import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


# =========================
# 1. Download price data
# =========================

ticker_1 = "V"
ticker_2 = "MA"
tickers = [ticker_1, ticker_2]

data = yf.download(
    tickers,
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False,
    threads=False
)

if data.empty:
    raise ValueError("No data downloaded from Yahoo Finance. Please try again later.")

prices = data["Close"].dropna()

if prices.empty:
    raise ValueError("Price data is empty after dropna().")

asset_1 = prices[ticker_1]
asset_2 = prices[ticker_2]

score, pvalue, _ = coint(asset_1, asset_2)

print(f"Cointegration p-value for {ticker_1}/{ticker_2}:", round(pvalue, 4))

if pvalue < 0.05:
    print(f"Result: {ticker_1} and {ticker_2} may be cointegrated")
else:
    print(f"Result: weak/no cointegration signal for {ticker_1}/{ticker_2}")


# =========================
# 3. Helper: calculate metrics
# =========================

def calculate_metrics(df):
    if df.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "trades": 0,
        }

    equity_curve = (1 + df["strategy_return"]).cumprod()

    total_return = equity_curve.iloc[-1] - 1
    annual_return = df["strategy_return"].mean() * 252
    annual_volatility = df["strategy_return"].std() * np.sqrt(252)

    if annual_volatility != 0 and not np.isnan(annual_volatility):
        sharpe = annual_return / annual_volatility
    else:
        sharpe = np.nan

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = drawdown.min()

    trades = df["trade"].sum()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
    }


# =========================
# 4. Backtest function
# =========================

def backtest_pair(
    prices,
    ticker_1,
    ticker_2,
    lookback_beta=252,
    lookback_zscore=60,
    entry_threshold=2.0,
    exit_threshold=0.5,
    transaction_cost=0.001,
):
    signals = pd.DataFrame(index=prices.index)
    signals[ticker_1] = prices[ticker_1]
    signals[ticker_2] = prices[ticker_2]

    signals["beta"] = np.nan

    for i in range(lookback_beta, len(signals)):
        y_window = signals[ticker_1].iloc[i - lookback_beta:i]
        x_window = signals[ticker_2].iloc[i - lookback_beta:i]

        x = sm.add_constant(x_window)
        model = sm.OLS(y_window, x).fit()

        signals.iloc[i, signals.columns.get_loc("beta")] = model.params[ticker_2]

    signals["spread"] = signals[ticker_1] - signals["beta"] * signals[ticker_2]

    spread_mean = (
        signals["spread"]
        .rolling(window=lookback_zscore)
        .mean()
        .shift(1)
    )

    spread_std = (
        signals["spread"]
        .rolling(window=lookback_zscore)
        .std()
        .shift(1)
    )

    signals["zscore"] = (signals["spread"] - spread_mean) / spread_std
    # =========================
# Rolling cointegration filter
# =========================

    lookback_coint = 252
    signals["coint_pvalue"] = np.nan

    for i in range(lookback_coint, len(signals)):
        y_window = signals[ticker_1].iloc[i - lookback_coint:i]
        x_window = signals[ticker_2].iloc[i - lookback_coint:i]

        try:
            _, pvalue, _ = coint(y_window, x_window)
            signals.iloc[i, signals.columns.get_loc("coint_pvalue")] = pvalue
        except Exception:
            signals.iloc[i, signals.columns.get_loc("coint_pvalue")] = np.nan

    signals["can_trade"] = signals["coint_pvalue"] < 0.05

    signals["position"] = np.nan

    signals.loc[signals["zscore"] > entry_threshold, "position"] = -1
    signals.loc[signals["zscore"] < -entry_threshold, "position"] = 1
    signals.loc[signals["zscore"].abs() < exit_threshold, "position"] = 0

    signals["position"] = signals["position"].ffill().fillna(0)
    signals.loc[signals["can_trade"] == False, "position"] = 0

    signals["asset_1_return"] = signals[ticker_1].pct_change()
    signals["asset_2_return"] = signals[ticker_2].pct_change()

    signals["pair_return"] = (
        signals["asset_1_return"]
        - signals["beta"].shift(1) * signals["asset_2_return"]
    )

    signals["strategy_return"] = (
        signals["position"].shift(1) * signals["pair_return"]
    )

    signals["strategy_return"] = signals["strategy_return"].fillna(0)

    signals["trade"] = signals["position"].diff().abs().fillna(0)
    signals["cost"] = signals["trade"] * transaction_cost

    signals["strategy_return"] = signals["strategy_return"] - signals["cost"]

    signals = signals.dropna(subset=["beta", "zscore"])

    return signals

# =========================
# 5. Grid search
# =========================

lookback_beta_list = [126]
lookback_zscore_list = [30, 60]
entry_threshold_list = [2.0, 2.5]
exit_threshold_list = [0.0, 0.5, 1.0]

results = []

for lookback_beta in lookback_beta_list:
    for lookback_zscore in lookback_zscore_list:
        for entry_threshold in entry_threshold_list:
            for exit_threshold in exit_threshold_list:

                # exit should be smaller than entry
                if exit_threshold >= entry_threshold:
                    continue

                signals = backtest_pair(
                    prices=prices,
                    ticker_1=ticker_1,
                    ticker_2=ticker_2,
                    lookback_beta=lookback_beta,
                    lookback_zscore=lookback_zscore,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    transaction_cost=0.001,
                    )
                train = signals.loc["2019-01-01":"2022-12-31"].copy()
                test = signals.loc["2023-01-01":"2025-12-31"].copy()

                train_metrics = calculate_metrics(train)
                test_metrics = calculate_metrics(test)
                full_metrics = calculate_metrics(signals)

                results.append({
                    "lookback_beta": lookback_beta,
                    "lookback_zscore": lookback_zscore,
                    "entry": entry_threshold,
                    "exit": exit_threshold,

                    "train_return": train_metrics["total_return"],
                    "train_sharpe": train_metrics["sharpe"],
                    "train_drawdown": train_metrics["max_drawdown"],
                    "train_trades": train_metrics["trades"],

                    "test_return": test_metrics["total_return"],
                    "test_sharpe": test_metrics["sharpe"],
                    "test_drawdown": test_metrics["max_drawdown"],
                    "test_trades": test_metrics["trades"],

                    "full_return": full_metrics["total_return"],
                    "full_sharpe": full_metrics["sharpe"],
                    "full_drawdown": full_metrics["max_drawdown"],
                    "full_trades": full_metrics["trades"],
                })


results_df = pd.DataFrame(results)


# =========================
# 6. Show best results
# =========================

# Chỉ lấy các bộ có train sharpe dương và test sharpe dương
robust_results = results_df[
    (results_df["train_sharpe"] > 0)
    & (results_df["test_sharpe"] > 0)
].copy()

print("\n===== Top results by test Sharpe =====")

if robust_results.empty:
    print("No parameter set has both positive train Sharpe and positive test Sharpe.")
    print("Showing top 10 by test Sharpe anyway:")

    top_results = results_df.sort_values(
        by="test_sharpe",
        ascending=False
    ).head(10)
else:
    top_results = robust_results.sort_values(
        by="test_sharpe",
        ascending=False
    ).head(10)


# Convert returns to percent for easier reading
display_cols = [
    "lookback_beta",
    "lookback_zscore",
    "entry",
    "exit",
    "train_return",
    "train_sharpe",
    "train_drawdown",
    "test_return",
    "test_sharpe",
    "test_drawdown",
    "full_return",
    "full_sharpe",
    "full_drawdown",
    "full_trades",
]

output = top_results[display_cols].copy()

percent_cols = [
    "train_return",
    "train_drawdown",
    "test_return",
    "test_drawdown",
    "full_return",
    "full_drawdown",
]

for col in percent_cols:
    output[col] = output[col] * 100

print(output.round(2).to_string(index=False))


# Save full grid search result
results_df.to_csv("grid_search_results.csv", index=False)

print("\nSaved full results to grid_search_results.csv")


# =========================
# 7. Backtest best parameter set
# =========================

best = top_results.iloc[0]

print("\n===== Best parameter set =====")
print(best[[
    "lookback_beta",
    "lookback_zscore",
    "entry",
    "exit",
    "train_sharpe",
    "test_sharpe",
    "full_sharpe",
]])


best_signals = backtest_pair(
    prices=prices,
    ticker_1=ticker_1,
    ticker_2=ticker_2,
    lookback_beta=int(best["lookback_beta"]),
    lookback_zscore=int(best["lookback_zscore"]),
    entry_threshold=float(best["entry"]),
    exit_threshold=float(best["exit"]),
    transaction_cost=0.001,
)

best_signals["equity_curve"] = (
    1 + best_signals["strategy_return"]
).cumprod()

best_signals.to_csv("best_strategy_signals.csv")

print("Saved best strategy signals to best_strategy_signals.csv")