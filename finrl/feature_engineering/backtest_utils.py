def apply_slippage_and_cost(price, action, slippage_pct=0.001, cost_pct=0.001, side="buy"):
    if side == "buy":
        exec_price = price * (1 + slippage_pct)
    else:
        exec_price = price * (1 - slippage_pct)
    cost = exec_price * abs(action) * cost_pct
    return exec_price, cost