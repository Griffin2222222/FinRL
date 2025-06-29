from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy import random as rd
from finrl.feature_engineering.contextual_reward import GPTContextualReward
import logging

gpt_contextual_reward = GPTContextualReward()


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
        stoploss_pct=0.05,  # 5% stoploss
        slippage_pct=0.001,  # 0.1% slippage
        latency_steps=1,  # 1 step latency
        volatility_window=20,  # for dynamic risk exposure
        max_volatility=0.03,  # max allowed volatility for full exposure
        min_exposure=0.2,  # min exposure at high volatility
        black_swan_thresh=0.15,  # 15% drop in one step triggers capital preservation
    ):
        self.macro_context = getattr(
            config, "macro_context", ""
        )  # or pass as config["macro_context"]
        self.gpt_client = getattr(
            config, "gpt_client", None
        )  # or pass as config["gpt_client"]
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # Advanced features
        self.stoploss_pct = stoploss_pct
        self.slippage_pct = slippage_pct
        self.latency_steps = latency_steps
        self.volatility_window = volatility_window
        self.max_volatility = max_volatility
        self.min_exposure = min_exposure
        self.black_swan_thresh = black_swan_thresh

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # For advanced features
        self.asset_history = []
        self.peak_asset = None
        self.price_history = []
        self.pending_actions = []

        # environment information
        self.env_name = "StockEnv"
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        # Advanced features
        self.asset_history = [self.total_asset]
        self.peak_asset = self.total_asset
        self.price_history = [price]
        self.pending_actions = []

        return self.get_state(price), {}  # state

    def step(self, actions):
        # === Latency modeling: queue actions ===
        self.pending_actions.append(actions)
        if len(self.pending_actions) <= self.latency_steps:
            actions = np.zeros_like(actions)
        else:
            actions = self.pending_actions.pop(0)

        # === Dynamic risk exposure adjustment ===
        exposure = 1.0
        sharpe = 0
        max_drawdown = 0
        volatility = 0
        if len(self.asset_history) > self.volatility_window:
            returns = (
                np.diff(self.asset_history[-self.volatility_window:]) / np.array(self.asset_history[-self.volatility_window:-1])
            )
            sharpe = returns.mean() / (returns.std() + 1e-9) if returns.std() > 0 else 0
            max_drawdown = (
                np.min(self.asset_history) / (self.peak_asset + 1e-9)
                if self.peak_asset else 0
            )
            volatility = returns.std()
            # Linearly scale exposure between 1 and min_exposure
            if volatility > self.max_volatility:
                exposure = self.min_exposure
            else:
                exposure = max(
                    self.min_exposure,
                    1 - (volatility / self.max_volatility) * (1 - self.min_exposure),
                )
        actions = actions * exposure

        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        # === Black swan/flash crash detection ===
        if len(self.price_history) > 0:
            price_drop = (self.price_history[-1] - price) / (self.price_history[-1] + 1e-9)
            black_swan = np.any(price_drop > self.black_swan_thresh)
            if black_swan:
                logging.warning("Black swan event detected: liquidating all positions.")
                self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
                self.stocks[:] = 0
                self.stocks_cool_down[:] = 0
                state = self.get_state(price)
                self.total_asset = self.amount
                self.asset_history.append(self.total_asset)
                self.price_history.append(price)
                reward = -10.0
                done = True
                self.episode_return = self.total_asset / self.initial_total_asset
                return state, reward, done, False, dict()

        # === Stoploss logic ===
        stoploss_trigger = price < (
            np.array(self.price_history[-1]) * (1 - self.stoploss_pct)
        )
        for idx in np.where(stoploss_trigger)[0]:
            if self.stocks[idx] > 0:
                sell_num_shares = self.stocks[idx]
                exec_price = price[idx] * (1 - self.slippage_pct)
                self.amount += exec_price * sell_num_shares * (1 - self.sell_cost_pct)
                self.stocks[idx] = 0
                self.stocks_cool_down[idx] = 0
                logging.info(f"Stoploss triggered for stock {idx} at price {price[idx]:.2f}")

        # === Trading logic with slippage and commission ===
        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)
            for index in np.where(actions < -min_action)[0]:
                if price[index] > 0:
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    exec_price = price[index] * (1 - self.slippage_pct)
                    self.stocks[index] -= sell_num_shares
                    self.amount += (
                        exec_price * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:
                if price[index] > 0:
                    exec_price = price[index] * (1 + self.slippage_pct)
                    buy_num_shares = min(self.amount // exec_price, actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= exec_price * buy_num_shares * (1 + self.buy_cost_pct)
                    self.stocks_cool_down[index] = 0
        else:
            exec_price = price * (1 - self.slippage_pct)
            self.amount += (self.stocks * exec_price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
            logging.info("Turbulence detected: liquidating all positions.")

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()

        # === Drawdown penalty ===
        self.peak_asset = max(self.peak_asset, total_asset)
        drawdown = (self.peak_asset - total_asset) / (self.peak_asset + 1e-9)
        reward = (total_asset - self.total_asset) * self.reward_scaling
        if drawdown > 0.05:
            penalty = np.exp(10 * (drawdown - 0.05))
            reward -= penalty
            logging.info(f"Drawdown penalty applied: {penalty:.2f}")
        # === Sharpe/volatility reward shaping ===
        if len(self.asset_history) > self.volatility_window:
            returns = (
                np.diff(self.asset_history[-self.volatility_window:]) / np.array(self.asset_history[-self.volatility_window:-1])
            )
            sharpe = returns.mean() / (returns.std() + 1e-9) if returns.std() > 0 else 0
            volatility = returns.std()
            reward += 0.1 * sharpe - 0.1 * abs(drawdown)
        reward = gpt_contextual_reward(
            reward, self.macro_context, actions, self.gpt_client
        )
        self.total_asset = total_asset
        self.asset_history.append(self.total_asset)
        self.price_history.append(price)
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset
        return state, reward, done, False, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
