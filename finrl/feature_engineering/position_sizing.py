import pandas as pd

def calc_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    df['ATR'] = atr
    return df

def kelly_criterion(win_rate, win_loss_ratio):
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    return max(kelly, 0)

def calc_dynamic_position(df, capital=1_000_000):
    win_rate = 0.55
    win_loss_ratio = 1.5
    kelly_fraction = kelly_criterion(win_rate, win_loss_ratio)
    df = calc_atr(df)
    df['position_size'] = capital * kelly_fraction / (df['ATR'] + 1e-9)
    df['position_size'] = df['position_size'].clip(upper=capital)
    return df