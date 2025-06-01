import pandas as pd

try:
    import talib
except ImportError:
    talib = None

def add_talib_indicators(df):
    if talib is None:
        raise ImportError("TA-Lib is not installed. Please install it to use TA-Lib indicators.")
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_hist'] = macdhist
    df['MACD_hist_div'] = df['MACD_hist'].diff()
    return df