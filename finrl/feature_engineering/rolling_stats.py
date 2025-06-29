def add_rolling_stats(df, window=20):
    returns = df["close"].pct_change().fillna(0)
    risk_free_rate = 0.0

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_downside_std = returns[returns < 0].rolling(window).std()
    sharpe = (rolling_mean - risk_free_rate) / (rolling_std + 1e-9)
    sortino = (rolling_mean - risk_free_rate) / (rolling_downside_std + 1e-9)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window, min_periods=1).max()
    drawdown = (cumulative - rolling_max) / rolling_max

    df["rolling_sharpe"] = sharpe
    df["rolling_sortino"] = sortino
    df["rolling_drawdown"] = drawdown
    return df
