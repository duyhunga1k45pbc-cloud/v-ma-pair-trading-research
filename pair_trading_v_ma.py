import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


# ============================================================
# 1. Download historical price data
# ============================================================

# Define the pair we want to research.
# ticker_1 is treated as the dependent asset.
# ticker_2 is used as the hedge asset.
ticker_1 = "V"
ticker_2 = "MA"
tickers = [ticker_1, ticker_2]

# Download adjusted historical price data from Yahoo Finance.
# auto_adjust=True adjusts prices for dividends and stock splits.
data = yf.download(
    tickers,
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False,
    threads=False
)

# Stop the program if Yahoo Finance returns no data.
if data.empty:
    raise ValueError("No data downloaded from Yahoo Finance. Please try again later.")

# Use adjusted close prices and remove missing rows.
prices = data["Close"].dropna()

if prices.empty:
    raise ValueError("Price data is empty after dropna().")

# Extract the two price series.
asset_1 = prices[ticker_1]
asset_2 = prices[ticker_2]


# ============================================================
# 2. Cointegration test
# ============================================================

# Test whether the two price series have a long-term statistical relationship.
# The null hypothesis of the cointegration test is:
# H0: the two assets are NOT cointegrated.
score, pvalue, _ = coint(asset_1, asset_2)

print(f"Cointegration p-value for {ticker_1}/{ticker_2}:", round(pvalue, 4))

# If p-value < 0.05, we reject the null hypothesis
# and consider the pair as potentially cointegrated.
if pvalue < 0.05:
    print(f"Result: {ticker_1} and {ticker_2} may be cointegrated")
else:
    print(f"Result: weak/no cointegration signal for {ticker_1}/{ticker_2}")


# ============================================================
# 3. Helper function: calculate performance metrics
# ============================================================

def calculate_metrics(df):
    """
    Calculate backtest performance metrics.

    Input:
        df: DataFrame containing strategy_return and trade columns.

    Output:
        Dictionary of performance metrics:
        - total_return
        - annual_return
        - annual_volatility
        - sharpe
        - max_drawdown
        - trades
    """

    # If the input dataframe is empty, return NaN values.
    if df.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "trades": 0,
        }

    # Equity curve shows how the strategy capital grows over time.
    equity_curve = (1 + df["strategy_return"]).cumprod()

    # Total return over the whole period.
    total_return = equity_curve.iloc[-1] - 1

    # Annualized return.
    # 252 is commonly used as the approximate number of trading days in a year.
    annual_return = df["strategy_return"].mean() * 252

    # Annualized volatility.
    annual_volatility = df["strategy_return"].std() * np.sqrt(252)

    # Sharpe ratio = annual return / annual volatility.
    # This measures return per unit of risk.
    if annual_volatility != 0 and not np.isnan(annual_volatility):
        sharpe = annual_return / annual_volatility
    else:
        sharpe = np.nan

    # Drawdown measures the decline from previous equity peak.
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = drawdown.min()

    # Count number of position changes.
    trades = df["trade"].sum()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
    }


# ============================================================
# 4. Backtest function
# ============================================================

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
    """
    Run a pair trading backtest.

    Parameters:
        prices:
            DataFrame of historical prices.

        ticker_1:
            First asset in the pair.
            The strategy goes long/short this asset against ticker_2.

        ticker_2:
            Second asset in the pair.
            This is used as the hedge asset.

        lookback_beta:
            Number of historical trading days used to estimate rolling beta.
            Example: 126 is about 6 months, 252 is about 1 trading year.

        lookback_zscore:
            Number of days used to calculate rolling spread mean/std.

        entry_threshold:
            Z-score level used to enter a trade.

        exit_threshold:
            Z-score level used to exit a trade.

        transaction_cost:
            Cost applied whenever the position changes.

    Output:
        signals DataFrame containing:
        - prices
        - rolling beta
        - spread
        - z-score
        - position
        - returns
        - transaction cost
        - strategy return
    """

    # Create a DataFrame to store all backtest data.
    signals = pd.DataFrame(index=prices.index)
    signals[ticker_1] = prices[ticker_1]
    signals[ticker_2] = prices[ticker_2]

    # ------------------------------------------------------------
    # 4.1 Rolling beta calculation
    # ------------------------------------------------------------

    # beta is the hedge ratio:
    # spread = ticker_1 - beta * ticker_2
    #
    # We calculate beta using only historical data.
    # This avoids look-ahead bias.
    signals["beta"] = np.nan

    for i in range(lookback_beta, len(signals)):
        # Use only past data for beta estimation.
        y_window = signals[ticker_1].iloc[i - lookback_beta:i]
        x_window = signals[ticker_2].iloc[i - lookback_beta:i]

        # Run OLS regression:
        # ticker_1 = alpha + beta * ticker_2
        x = sm.add_constant(x_window)
        model = sm.OLS(y_window, x).fit()

        # Store rolling beta for the current day.
        signals.iloc[i, signals.columns.get_loc("beta")] = model.params[ticker_2]

    # ------------------------------------------------------------
    # 4.2 Spread and z-score calculation
    # ------------------------------------------------------------

    # Spread measures relative mispricing between the two assets.
    signals["spread"] = signals[ticker_1] - signals["beta"] * signals[ticker_2]

    # Calculate rolling mean of spread.
    # shift(1) ensures today's signal only uses information available up to yesterday.
    spread_mean = (
        signals["spread"]
        .rolling(window=lookback_zscore)
        .mean()
        .shift(1)
    )

    # Calculate rolling standard deviation of spread.
    spread_std = (
        signals["spread"]
        .rolling(window=lookback_zscore)
        .std()
        .shift(1)
    )

    # Z-score tells how far the spread is from its recent mean.
    signals["zscore"] = (signals["spread"] - spread_mean) / spread_std

    # ------------------------------------------------------------
    # 4.3 Rolling cointegration filter
    # ------------------------------------------------------------

    # This filter checks whether the pair is still cointegrated
    # based on recent historical data.
    #
    # If the recent p-value is below 0.05, trading is allowed.
    # Otherwise, the strategy stays out of the market.
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

    # Only allow trading when recent cointegration is strong enough.
    signals["can_trade"] = signals["coint_pvalue"] < 0.05

    # ------------------------------------------------------------
    # 4.4 Trading signal generation
    # ------------------------------------------------------------

    # position meaning:
    #  1  = long ticker_1, short ticker_2
    # -1  = short ticker_1, long ticker_2
    #  0  = no position
    signals["position"] = np.nan

    # If z-score is very high:
    # ticker_1 is expensive relative to ticker_2.
    # Therefore, short ticker_1 and long ticker_2.
    signals.loc[signals["zscore"] > entry_threshold, "position"] = -1

    # If z-score is very low:
    # ticker_1 is cheap relative to ticker_2.
    # Therefore, long ticker_1 and short ticker_2.
    signals.loc[signals["zscore"] < -entry_threshold, "position"] = 1

    # Exit the position when the spread moves back near its mean.
    signals.loc[signals["zscore"].abs() < exit_threshold, "position"] = 0

    # Forward-fill position so that the strategy keeps holding
    # until a new entry or exit signal appears.
    signals["position"] = signals["position"].ffill().fillna(0)

    # If recent cointegration is weak, force position to 0.
    signals.loc[signals["can_trade"] == False, "position"] = 0

    # ------------------------------------------------------------
    # 4.5 Return calculation
    # ------------------------------------------------------------

    # Daily percentage returns for both assets.
    signals["asset_1_return"] = signals[ticker_1].pct_change()
    signals["asset_2_return"] = signals[ticker_2].pct_change()

    # Pair return:
    # long ticker_1 and short beta * ticker_2.
    #
    # beta.shift(1) is used to avoid look-ahead bias.
    signals["pair_return"] = (
        signals["asset_1_return"]
        - signals["beta"].shift(1) * signals["asset_2_return"]
    )

    # Strategy return:
    # yesterday's position earns today's pair return.
    #
    # position.shift(1) avoids using today's signal to earn today's return.
    signals["strategy_return"] = (
        signals["position"].shift(1) * signals["pair_return"]
    )

    signals["strategy_return"] = signals["strategy_return"].fillna(0)

    # ------------------------------------------------------------
    # 4.6 Transaction cost
    # ------------------------------------------------------------

    # trade measures how much the position changed.
    # Example:
    # 0 -> 1  means opening a position.
    # 1 -> 0  means closing a position.
    # 1 -> -1 means reversing direction.
    signals["trade"] = signals["position"].diff().abs().fillna(0)

    # Apply transaction cost whenever position changes.
    signals["cost"] = signals["trade"] * transaction_cost

    # Subtract transaction costs from strategy return.
    signals["strategy_return"] = signals["strategy_return"] - signals["cost"]

    # Remove rows where beta or z-score is not ready yet.
    signals = signals.dropna(subset=["beta", "zscore"])

    return signals


# ============================================================
# 5. Grid search
# ============================================================

# Try different combinations of parameters.
# This helps evaluate whether the strategy is robust.
lookback_beta_list = [126]
lookback_zscore_list = [30, 60]
entry_threshold_list = [2.0, 2.5]
exit_threshold_list = [0.0, 0.5, 1.0]

results = []

for lookback_beta in lookback_beta_list:
    for lookback_zscore in lookback_zscore_list:
        for entry_threshold in entry_threshold_list:
            for exit_threshold in exit_threshold_list:

                # Exit threshold should be smaller than entry threshold.
                # Otherwise, the strategy logic does not make sense.
                if exit_threshold >= entry_threshold:
                    continue

                # Run backtest for this parameter combination.
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

                # Split data into train and test periods.
                # Train period is used for research and parameter evaluation.
                # Test period checks out-of-sample performance.
                train = signals.loc["2019-01-01":"2022-12-31"].copy()
                test = signals.loc["2023-01-01":"2025-12-31"].copy()

                # Calculate metrics for train, test, and full period.
                train_metrics = calculate_metrics(train)
                test_metrics = calculate_metrics(test)
                full_metrics = calculate_metrics(signals)

                # Store results.
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


# Convert result list to DataFrame for sorting and exporting.
results_df = pd.DataFrame(results)


# ============================================================
# 6. Show best results
# ============================================================

# Robust means the strategy performs positively
# in both train and test periods.
robust_results = results_df[
    (results_df["train_sharpe"] > 0)
    & (results_df["test_sharpe"] > 0)
].copy()

print("\n===== Top results by test Sharpe =====")

if robust_results.empty:
    # If no robust result is found, show top results by test Sharpe only.
    # These are for analysis only and should not be considered reliable.
    print("No parameter set has both positive train Sharpe and positive test Sharpe.")
    print("Showing top 10 by test Sharpe anyway:")

    top_results = results_df.sort_values(
        by="test_sharpe",
        ascending=False
    ).head(10)
else:
    # If robust results exist, sort them by test Sharpe.
    top_results = robust_results.sort_values(
        by="test_sharpe",
        ascending=False
    ).head(10)


# Select columns to display.
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

# Convert return and drawdown columns from decimal to percentage.
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

# Print final table in terminal.
print(output.round(2).to_string(index=False))


# Save full grid search results to CSV.
results_df.to_csv("grid_search_results.csv", index=False)

print("\nSaved full results to grid_search_results.csv")


# ============================================================
# 7. Backtest best parameter set
# ============================================================

# Pick the first row from the top results as the best candidate.
# Note:
# If no robust result was found, this is only the best by test Sharpe,
# not necessarily a reliable trading strategy.
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

# Run the backtest again using the selected best parameter set.
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

# Calculate equity curve for the selected best strategy.
best_signals["equity_curve"] = (
    1 + best_signals["strategy_return"]
).cumprod()

# Save full signal table for later analysis.
best_signals.to_csv("best_strategy_signals.csv")

print("Saved best strategy signals to best_strategy_signals.csv")