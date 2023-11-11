import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Any
import random


class AssetTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            data_df: pd.DataFrame,
            initial_balance: float = 100_000.00,
            render_mode: str = 'human'
            ) -> None:
        super(AssetTradingEnv, self).__init__()
        self.data_df = data_df.copy()
        self.positions = [-1, 0, 1]
        self.sell = self.positions[0]
        self.hold = self.positions[1]
        self.buy = self.positions[2]
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        self.data_df['date'] = self.data_df.index
        self._features_cols = [col for col in self.data_df.columns
                               if 'feature' in col]
        info_cols = set(self.data_df.columns) - set(self._features_cols)
        extras_cols = {key: index for index, key in enumerate(info_cols)}
        extras_array = np.array(self.data_df[extras_cols.keys()])
        self.history_info_obj = HistoryInfo(extras_cols, extras_array)

        self._step = 0
        self._initial_step = self._step
        self._termination_balance = self.initial_balance * 0.05
        self._max_episode_steps = len(self.data_df) - 1
        self._obs_array = np.array(self.data_df[self._features_cols],
                                   dtype=np.float32)
        self._asset_price_array = np.array(self.data_df['close'])

        shape = len(self._features_cols)
        self.observation_space =\
            spaces.Box(low=-1, high=1,
                       shape=(shape,))
        self.action_space = spaces.Discrete(len(self.positions))

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed, options=options)
        # self.current_trade_position = self.np_random.choice(self.positions)
        # If randomized, ensure purchase price is updated correctly.

        self.current_trade_position = 0.
        self._step = 0
        self.history_info_obj.add_info(
            step=self._step,
            signal=0,
            portfolio_balance=self.initial_balance,
            available_funds=self.initial_balance,
            unrealized_trade=0.,
            position=0.,
            trade_duration=0,
            purchase_close_price=0.,
            step_reward=0.,
            total_reward=0.,
            risk_value=0.)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        signal = self.positions[action]
        self._step += 1

        portfolio_balance, available_funds, unrealized_trade, position, \
            trade_duration, purchase_close_price = \
            self._update_portfolio(signal)

        risk_value = 0.
        total_reward = self.history_info_obj.get_step_and_col(
            self._step-1, 'total_reward')
        step_reward = self.calc_reward(portfolio_balance)
        total_reward += step_reward

        self.history_info_obj.add_info(
            step=self._step,
            signal=signal,
            portfolio_balance=portfolio_balance,
            available_funds=available_funds,
            unrealized_trade=unrealized_trade,
            position=position,
            trade_duration=trade_duration,
            purchase_close_price=purchase_close_price,
            step_reward=step_reward,
            total_reward=total_reward,
            risk_value=risk_value)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = False if portfolio_balance > self._termination_balance\
            else True
        truncated = False if self._step < self._max_episode_steps else True
        info = self._get_info()

        if terminated or truncated:
            self.render()

        # Use for troubleshooting reward and risk values.
        # print()
        # print(f"Signal: {signal}")
        # for key, value in info.items():
        #     print(key, ": ", value)
        # if self._step == 10:
        #     return observation, reward, True, True, info
        # print(info)
        # print()
        return observation, reward, terminated, truncated, info

    def render(self):
        m_final = self.history_info_obj.get_step_and_col(
            self._step, 'close')
        m_initial = self.history_info_obj.get_step_and_col(
            self._initial_step, 'close')
        m_return = (m_final - m_initial) / m_initial * 100

        p_final = self.history_info_obj.get_step_and_col(
            self._step, 'portfolio_balance')
        p_initial = self.history_info_obj.get_step_and_col(
            self._initial_step, 'portfolio_balance')
        p_return = (p_final - p_initial) / p_initial * 100

        print(f"|  Market Return:{m_return:9.2f}% |",
              f"  Portfolio Return:{p_return:9.2f}% |")

    def close(self):
        pass

    def _update_portfolio(self, action: int):
        previous_step = self._step - 1
        portfolio_balance = self.history_info_obj.get_step_and_col(
            previous_step, 'portfolio_balance')
        available_funds = self.history_info_obj.get_step_and_col(
            previous_step, 'available_funds')
        unrealized_trade = self.history_info_obj.get_step_and_col(
            previous_step, 'unrealized_trade')
        purchase_close_price = self.history_info_obj.get_step_and_col(
            previous_step, 'purchase_close_price')
        position = self.history_info_obj.get_step_and_col(
            previous_step, 'position')
        trade_duration = self.history_info_obj.get_step_and_col(
            previous_step, 'trade_duration')
        _close_price = self._asset_price_array[self._step]

        if position >= 1:
            if action == self.hold or action == self.buy:
                unrealized_trade = round(_close_price * position, 3)
                portfolio_balance = round(available_funds
                                          + unrealized_trade, 3)
                trade_duration += 1
            if action == self.sell:
                unrealized_trade = round(_close_price * position, 3)
                portfolio_balance = round(available_funds
                                          + unrealized_trade, 3)
                available_funds = round(available_funds + unrealized_trade, 3)
                unrealized_trade = 0.
                position = 0.
                trade_duration = 0
                purchase_close_price = 0.
        elif action == self.buy and position == 0:
            unrealized_trade = round(available_funds, 3)
            purchase_close_price = round(_close_price, 3)
            position = round(unrealized_trade / purchase_close_price, 3)
            available_funds = round(available_funds - unrealized_trade, 3)
            portfolio_balance = round(available_funds + unrealized_trade, 3)

        return portfolio_balance, available_funds, unrealized_trade, \
            position, trade_duration, purchase_close_price

    def _get_obs(self):
        obs = self._obs_array[self._step]
        return obs

    def _get_info(self):
        info = self.history_info_obj.get_step(self._step)
        return info

    def _get_reward(self):
        return self.history_info_obj.get_step_and_col(self._step,
                                                      'step_reward')

    def calc_reward(self, p_current: float) -> float:
        """
        Function for calculating drawdown

        Args:
            History
            Window Size

        Returns:
            risk weighted positive profit - risk weighted drawdown
        """
        # pull data from history
        previous_step = self._step - 1
        p_previous = self.history_info_obj.get_step_and_col(
            previous_step, 'portfolio_balance')
        reward = (p_current - p_previous) / p_previous
        # print(reward)
        # position = self.history_info_obj.get_step_and_col(
        #     previous_step, 'position')
        # random.seed(42)
        # reward = random.randrange(-100, 100)/100
        return reward


class HistoryInfo():
    def __init__(self, extras_cols: dict, extras_array: np.array) -> None:
        self._extras_cols = extras_cols
        self._extras_array = extras_array
        self._history_info_dict = dict()

    def get_extras_data_col(self, step: int, col_name: str):
        row_index = self._extras_cols[col_name]
        return self._extras_array[step, row_index]

    def add_info(
            self,
            step: int,
            signal: int,
            portfolio_balance: float,
            available_funds: float,
            unrealized_trade: float,
            position: float,
            trade_duration: int,
            purchase_close_price: float,
            step_reward: float,
            total_reward: float,
            risk_value: float
            ) -> None:

        date = self.get_extras_data_col(step, 'date')
        close = self.get_extras_data_col(step, 'close')
        open = self.get_extras_data_col(step, 'open')
        low = self.get_extras_data_col(step, 'low')
        high = self.get_extras_data_col(step, 'high')
        volume = self.get_extras_data_col(step, 'volume')

        step_info = {'step': step,
                     'date': date,
                     'signal': signal,
                     'portfolio_balance': portfolio_balance,
                     'available_funds': available_funds,
                     'unrealized_trade': unrealized_trade,
                     'position': position,
                     'trade_duration': trade_duration,
                     'purchase_close_price': purchase_close_price,
                     'step_reward': step_reward,
                     'total_reward': total_reward,
                     'risk_value': risk_value,
                     'close': close,
                     'open': open,
                     'low': low,
                     'high': high,
                     'volume': volume}

        atr_indices = [key for key in self._extras_cols.keys() if 'atr' in key]
        for atr_label in atr_indices:
            atr = self.get_extras_data_col(step, atr_label)
            step_info[atr_label] = atr

        v_indices = [key for key in self._extras_cols.keys() if '_v_' in key]
        for v_label in v_indices:
            v = self.get_extras_data_col(step, v_label)
            step_info[v_label] = v

        a_indices = [key for key in self._extras_cols.keys() if '_a_' in key]
        for a_label in a_indices:
            a = self.get_extras_data_col(step, a_label)
            step_info[a_label] = a

        self._history_info_dict[step] = step_info

    def get_step(self, step: int) -> dict:
        return self._history_info_dict[step]

    def get_step_and_col(self, step: int, col_name: str) -> Any:
        step_dict = self.get_step(step)
        return step_dict[col_name]


if __name__ == '__main__':
    from data_processor import DataProcessor
    dp = DataProcessor()
    symbol = 'TQQQ'
    start_date = '2010-02-11'
    stop_date = '2023-11-30'
    tqqq = dp.download_data_df_from_yf(symbol, start_date, stop_date)
    preprocessed_df = dp.preprocess_data(tqqq)
    env = AssetTradingEnv(preprocessed_df)
    obs, info = env.reset()
    random.seed(1)
    for index in range(15):
        action = random.action([-1, 0, 1])
        print()
        print(f"Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        for key, value in info.items():
            print(key, ": ", value)
