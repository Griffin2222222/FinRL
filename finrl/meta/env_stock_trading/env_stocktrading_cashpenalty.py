from __future__ import annotations

import random
import time
from copy import deepcopy
import logging

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from finrl.feature_engineering.backtest_utils import apply_slippage_and_cost
from finrl.feature_engineering.sentiment import get_sentiment_score  # Ensure this exists

matplotlib.use("Agg")


class StockTradingEnvCashpenalty(gym.Env):
    """
    A stock trading environment for OpenAI gym
    This environment penalizes the model for not maintaining a reserve of cash.
    This enables the model to manage cash reserves in addition to performing trading procedures.
    Reward at any step is given as follows
        r_i = (sum(cash, asset_value) - initial_cash - max(0, sum(cash, asset_value)*cash_penalty_proportion-cash))/(days_elapsed)
        This reward function takes into account a liquidity requirement, as well as long-term accrued rewards.
    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        buy_cost_pct (float): cost for buying shares
        sell_cost_pct (float): cost for selling shares
        hmax (int, array): maximum cash to be traded in each trade per asset. If an array is provided, then each index correspond to each asset
        discrete_actions (bool): option to choose whether perform dicretization on actions space or not
        shares_increment (int): multiples number of shares can be bought in each trade. Only applicable if discrete_actions=True
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        initial_amount: (int, float): Amount of cash initially available
        daily_information_columns (list(str)): Columns to use when building state space from the dataframe. It could be OHLC columns or any other variables such as technical indicators and turbulence index
        cash_penalty_proportion (int, float): Penalty to apply if the algorithm runs out of cash
        patient (bool): option to choose whether end the cycle when we're running out of cash or just don't buy anything until we got additional cash

    RL Inputs and Outputs
        action space: [<n_assets>,] in range {-1, 1}
        state space: {start_cash, [shares_i for in in assets], [[indicator_j for j in indicators] for i in assets]]}
    TODO:
        Organize functions
        Write README
        Document tests
    """

    metadata = {"render.modes": ["human"]}

    # === RISK CONTROL DEFAULTS ===
    # Set turbulence, stop-loss, and profit thresholds for robust risk management
    DEFAULT_TURBULENCE_THRESHOLD = 100
    DEFAULT_STOPLOSS_PENALTY = 0.9  # Sell if price < 90% of avg buy
    DEFAULT_PROFIT_TAKE_RATIO = 1.2  # Take profit if price > 120% of avg buy
    DEFAULT_VIX_THRESHOLD = 30  # Example: avoid trading if VIX > 30

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        turbulence_threshold=None,
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
        volatility_window=20,
        max_volatility=0.03,
        min_exposure=0.2,
        black_swan_thresh=0.15,
        macro_context=None,
        gpt_client=None,
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        self.volatility_window = volatility_window
        self.max_volatility = max_volatility
        self.min_exposure = min_exposure
        self.black_swan_thresh = black_swan_thresh
        self.asset_history = []
        self.peak_asset = None
        self.price_history = []
        self.pending_actions = []
        self.macro_context = macro_context
        self.gpt_client = gpt_client
        if (
            "sentiment_score" in df.columns
            and "sentiment_score" not in daily_information_cols
        ):
            daily_information_cols = daily_information_cols + ["sentiment_score"]
        self.df = df
        if self.cache_indicator_data:
            print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("data cached!")

    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    @property
    def cash_on_hand(self):
        # amount of cash held at current timestep
        return self.state_memory[-1][0]

    @property
    def holdings(self):
        # Quantity of shares held at current timestep
        return self.state_memory[-1][1 : len(self.assets) + 1]

    @property
    def closings(self):
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.seed()
        self.sum_trades = 0
        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)
        self.asset_history = [self.initial_amount]
        self.peak_asset = self.initial_amount
        self.price_history = [self.get_date_vector(self.date_index, cols=["close"])]
        self.pending_actions = []
        return init_state

    def get_date_vector(self, date, cols=None):
        if (cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        # Add outputs to logger interface
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
        logger.record(
            "environment/total_assets",
            int(self.account_information["total_assets"][-1]),
        )
        reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
        logger.record("environment/total_trades", self.sum_trades)
        logger.record(
            "environment/avg_daily_trades",
            self.sum_trades / (self.current_step),
        )
        logger.record(
            "environment/avg_daily_trades_per_asset",
            self.sum_trades / (self.current_step) / len(self.assets),
        )
        logger.record("environment/completed_steps", self.current_step)
        logger.record(
            "environment/sum_rewards", np.sum(self.account_information["reward"])
        )
        logger.record(
            "environment/cash_proportion",
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1],
        )
        return state, reward, True, {}

    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))
        # Log to file for audit
        logging.basicConfig(
            filename="trades_log.csv", level=logging.INFO, format="%(message)s"
        )
        logging.info(",".join(map(str, rec)))

    def log_header(self):
        if self.printed_header is False:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD_unsc",
                    "GAINLOSS_PCT",
                    "CASH_PROPORTION",
                )
            )
            self.printed_header = True

    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            assets = self.account_information["total_assets"][-1]
            cash = self.account_information["cash"][-1]
            cash_penalty = max(0, (assets * self.cash_penalty_proportion - cash))
            assets -= cash_penalty
            # Add drawdown penalty
            peak = max(self.account_information["total_assets"])
            drawdown = (peak - assets) / (peak + 1e-9)
            # Add risk-adjusted reward (Sharpe proxy)
            rewards = self.account_information["reward"]
            sharpe = np.mean(rewards) / (np.std(rewards) + 1e-9) if len(rewards) > 1 else 0
            reward = (assets / self.initial_amount) - 1
            reward /= self.current_step
            # Penalize large drawdowns, reward high Sharpe
            reward += 0.05 * sharpe - 0.1 * drawdown
            return reward

    def get_transactions(self, actions):
        """
        This function takes in a raw 'action' from the model and makes it into realistic transactions
        This function includes logic for discretizing
        It also includes turbulence logic.
        """
        # record actions of the model
        self.actions_memory.append(actions)

        # multiply actions by the hmax value
        actions = actions * self.hmax

        # Do nothing for shares with zero value
        actions = np.where(self.closings > 0, actions, 0)

        # discretize optionally
        if self.discrete_actions:
            # convert into integer because we can't buy fraction of shares
            actions = actions // self.closings
            actions = actions.astype(int)
            # round down actions to the nearest multiplies of shares_increment
            actions = np.where(
                actions >= 0,
                (actions // self.shares_increment) * self.shares_increment,
                ((actions + self.shares_increment) // self.shares_increment)
                * self.shares_increment,
            )
        else:
            actions = actions / self.closings

        # can't sell more than we have
        actions = np.maximum(actions, -np.array(self.holdings))

        # deal with turbulence
        if self.turbulence_threshold is not None:
            # if turbulence goes over threshold, just clear out all positions
            if self.turbulence >= self.turbulence_threshold:
                actions = -(np.array(self.holdings))
                self.log_step(reason="TURBULENCE")

        return actions

    def step(self, actions):
        # === Latency modeling: queue actions ===
        self.pending_actions.append(actions)
        if len(self.pending_actions) <= 1:
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

        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())
        else:
            """
            First, we need to compute values of holdings, save these, and log everything.
            Then we can reward our model for its earnings.
            """
            # compute value of cash + assets
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            asset_value = np.dot(self.holdings, self.closings)
            # log the values of cash, assets, and total assets
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(asset_value)
            self.account_information["total_assets"].append(begin_cash + asset_value)

            # compute reward once we've computed the value of things!
            reward = self.get_reward()
            self.account_information["reward"].append(reward)

            # Now, let's get down to business at hand.
            transactions = self.get_transactions(actions)

            # compute our proceeds from sells, and add to cash
            sells = -np.clip(transactions, -np.inf, 0)
            proceeds = 0
            sell_costs = 0
            for i, sell_amt in enumerate(sells):
                if sell_amt > 0:
                    exec_price, cost = apply_slippage_and_cost(
                        self.closings[i],
                        sell_amt,
                        slippage_pct=0.001,
                        cost_pct=self.sell_cost_pct,
                        side="sell",
                    )
                    proceeds += exec_price * sell_amt
                    sell_costs += cost
            coh = begin_cash + proceeds
            # compute the cost of our buys
            buys = np.clip(transactions, 0, np.inf)
            spend = 0
            buy_costs = 0
            for i, buy_amt in enumerate(buys):
                if buy_amt > 0:
                    exec_price, cost = apply_slippage_and_cost(
                        self.closings[i],
                        buy_amt,
                        slippage_pct=0.001,
                        cost_pct=self.buy_cost_pct,
                        side="buy",
                    )
                    spend += exec_price * buy_amt
                    buy_costs += cost
            costs = sell_costs + buy_costs
            # if we run out of cash...
            if (spend + costs) > coh:
                if self.patient:
                    # ... just don't buy anything until we got additional cash
                    self.log_step(reason="CASH SHORTAGE")
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    # ... end the cycle and penalize
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )
            self.transaction_memory.append(
                transactions
            )  # capture what the model's could do
            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh
            # update our holdings
            coh = coh - spend - costs
            holdings_updated = self.holdings + transactions
            self.date_index += 1
            if self.turbulence_threshold is not None:
                self.turbulence = self.get_date_vector(
                    self.date_index, cols=["turbulence"]
                )[0]
            # Update State
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            # Liquidity penalty: penalize if buy/sell volume exceeds a fraction of average volume
            liquidity_penalty = 0
            avg_volumes = np.array(
                self.get_date_vector(self.date_index, cols=["volume"])
            )
            for i, (buy_amt, sell_amt) in enumerate(zip(buys, sells)):
                # Assume 10% of avg volume is max reasonable trade size
                max_liq = avg_volumes[i] * 0.1
                if buy_amt > max_liq:
                    liquidity_penalty -= (buy_amt - max_liq) / max_liq
                if sell_amt > max_liq:
                    liquidity_penalty -= (sell_amt - max_liq) / max_liq
            reward += liquidity_penalty

            # Black swan/flash crash detection
            price = self.get_date_vector(self.date_index, cols=["close"])
            price_drop = (self.price_history[-1] - price) / (self.price_history[-1] + 1e-9)
            black_swan = np.any(price_drop > self.black_swan_thresh)
            if black_swan:
                # Capital preservation: liquidate all positions, hold cash
                begin_cash = self.cash_on_hand
                holdings = self.holdings
                proceeds = np.dot(holdings, price) * (1 - self.sell_cost_pct)
                coh = begin_cash + proceeds
                holdings_updated = np.zeros_like(holdings)
                self.state_memory.append([coh] + list(holdings_updated) + self.get_date_vector(self.date_index))
                self.asset_history.append(coh)
                self.price_history.append(price)
                reward = -10.0  # Large penalty for black swan event
                self.log_step(reason="BLACK SWAN")
                return self.state_memory[-1], reward, True, {}

            # After updating holdings, update asset history and price history
            total_asset = self.state_memory[-1][0] + np.dot(self.state_memory[-1][1:len(self.assets)+1], price)
            self.asset_history.append(total_asset)
            self.price_history.append(price)
            self.peak_asset = max(self.peak_asset, total_asset)
            drawdown = (self.peak_asset - total_asset) / (self.peak_asset + 1e-9)
            reward = self.get_reward()
            if drawdown > 0.05:
                reward -= np.exp(10 * (drawdown - 0.05))
            # Add rolling Sharpe and drawdown bonuses/penalties
            reward += 0.1 * sharpe - 0.1 * abs(max_drawdown)
            # Sentiment-based reward shaping (if available)
            if self.macro_context is not None and self.gpt_client is not None:
                try:
                    sentiment = get_sentiment_score(self.macro_context, self.gpt_client)
                    reward += 0.05 * sentiment
                except Exception as e:
                    logging.warning(f"Sentiment API error: {e}")

            # Advanced logging for explainability
            logging.info(f"step={self.current_step}, asset={total_asset}, reward={reward}, exposure={exposure}, sharpe={sharpe}, drawdown={drawdown}")
            # Save state-action pairs for SHAP/surrogate explainability
            if not hasattr(self, 'explain_log'):
                self.explain_log = []
            self.explain_log.append({'state': self.state_memory[-1], 'action': actions, 'reward': reward})

            return state, reward, False, {}

    def get_sb_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]) :
            ]
            return pd.DataFrame(self.account_information)

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]) :],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,
                }
            )
