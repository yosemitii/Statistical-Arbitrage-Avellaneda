# Intro

**Paper Link:** https://math.nyu.edu/~avellane/AvellanedaLeeStatArb071108.pdf
- **Stock daily file**: CRSP Stock Daily File v2 (CIZ)
- **SPY 500 holdings**: CRSP Mutual Fund File


## How to run

### 1. Data Source Settings
Method1: 
- create a `.env` file in the working directory
- Save this content in `.env` file, replace it with your own path
```angular2html
STOCK_DATA_PATH="STOCK_FILE_PATH_HERE"
SPY_HOLDINGS_PATH="STOCK_SPY_HOLDING_FILE_PATH_HERE"
```
Method 2:
Just change the paths in `backtesting.py` code to your paths

### 2. Run
Run the main function of `backtesting.py`
