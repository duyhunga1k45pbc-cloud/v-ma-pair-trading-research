# V/MA Pair Trading Research

This project is a quantitative research and backtesting study of a pair trading strategy between **Visa (V)** and **Mastercard (MA)** using Python.

The goal is to test whether two stocks from the same industry have a stable long-term relationship and whether deviations in their price spread can be used to generate long/short trading signals.

## Project Overview

Visa and Mastercard are both major payment network companies. Because they operate in a similar industry, their stock prices may move together over time.

This project tests that relationship using:

- Cointegration testing
- OLS hedge ratio estimation
- Rolling beta
- Spread z-score
- Long/short trading signals
- Transaction cost
- Train/test validation
- Grid search for strategy parameters

## Strategy Logic

The strategy is based on the spread between Visa and Mastercard:

```text
spread = V - beta * MA
```

Where:

- `V` is the adjusted close price of Visa
- `MA` is the adjusted close price of Mastercard
- `beta` is the hedge ratio estimated by OLS regression

A z-score is calculated from the spread to measure how far the current spread is from its recent historical mean.

Trading rules:

```text
If z-score > entry threshold:
    Short V and long MA

If z-score < -entry threshold:
    Long V and short MA

If abs(z-score) < exit threshold:
    Close the position
```

## Data Source

Historical adjusted close prices are downloaded from Yahoo Finance using the `yfinance` Python library.

The main test period is from 2018 to 2025.

## Cointegration Test

The first step is to test whether Visa and Mastercard are cointegrated.

The initial cointegration test result for V/MA showed:

```text
p-value ≈ 0.0014
```

Since the p-value is below 0.05, the result suggests that Visa and Mastercard may have a long-term statistical relationship.

However, cointegration alone does not guarantee that a trading strategy will be profitable.

## Avoiding Look-Ahead Bias

The first version of the model used a fixed hedge ratio calculated from the full dataset.

This was improved by using **rolling beta**, where the hedge ratio at each point in time is calculated using only historical data available before that point.

This helps avoid look-ahead bias.

The model also uses:

```python
position.shift(1)
```

to make sure that today's signal is only applied to the next trading period.

## Backtest Metrics

The strategy is evaluated using:

- Total Return
- Annual Return
- Annual Volatility
- Sharpe Ratio
- Max Drawdown
- Number of Trades
- Yearly Returns

## Train/Test Split

To evaluate robustness, the data is split into:

```text
Train period: 2019–2022
Test period: 2023–2025
```

The train period is used to evaluate parameter behavior, while the test period is used to check whether the strategy still works outside the original testing period.

## Grid Search

A grid search is implemented to test different combinations of:

- `lookback_beta`
- `lookback_zscore`
- `entry_threshold`
- `exit_threshold`

The model compares train Sharpe, test Sharpe, return, drawdown, and number of trades.

## Key Findings

Visa and Mastercard showed a strong cointegration signal over the full period.

However, after applying rolling beta, transaction costs, and train/test validation, the simple z-score pair trading strategy was not robust enough.

Some parameter sets performed well in either the train period or the test period, but the grid search did not find a parameter set that consistently produced positive Sharpe ratios across both periods.

## Conclusion

The research shows that:

- V and MA may be cointegrated.
- A spread z-score pair trading strategy can be built and backtested.
- Rolling beta is important to reduce look-ahead bias.
- Train/test validation is necessary to avoid overfitting.
- The current strategy logic is not robust enough for practical trading.

This project is mainly a learning and research project, not a live trading strategy.

## Possible Improvements

Future improvements could include:

- Rolling cointegration filter
- Regime detection
- Better position sizing
- Stop-loss logic
- Testing more stock pairs
- Long/short equity portfolio strategy
- More realistic transaction cost and slippage modeling

## Project Structure

```text
.
├── pair_trading_v_ma.py
├── requirements.txt
├── grid_search_results.csv
├── best_strategy_signals.csv
├── zscore_v_ma.png
├── equity_curve_v_ma.png
├── README.md
└── .gitignore
```

## How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backtest:

```bash
python pair_trading_v_ma.py
```

The script will generate:

```text
grid_search_results.csv
best_strategy_signals.csv
zscore_v_ma.png
equity_curve_v_ma.png
```

## Disclaimer

This project is for educational and research purposes only. It is not financial advice and should not be used as a live trading system without further validation.
