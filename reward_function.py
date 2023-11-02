import pandas as pd
import numpy as np
from gym_trading_env.utils.history import History


def reward_function_drawdown(history: History, win_size: int = 144) -> float:
        """
        Reward Function

        Args:
            History
            Window Size

        Returns:
            risk weighted positive profit - risk weighted drawdown
        """
        # 0 to 1 to adjust risk tolerance
        profit_reward_weight = 0.5

        # pull data from history
        data_close = np.array(history['data_close'][-win_size:], dtype=np.float64)
        pos = np.array(history['position'][-win_size:], dtype=np.float64)

        # calculate log returns for the rolling window
        log_data = np.log(data_close)
        log_return = np.where(pos[1:] == 1, np.diff(log_data), 0)

        # Calculate drawdown for the rolling window
        portfolio_value = np.exp(log_return).cumprod()
        peak_value = np.maximum.accumulate(portfolio_value)
        drawdown = 1.0 - (portfolio_value / peak_value)

        # Calculate the mean drawdown
        mean_drawdown = np.mean(drawdown)

        # Calculate positive profit as the total profit in the window
        positive_profit = np.sum(np.where(log_return > 0, log_return, 0))

        # Combine drawdown and positive profit in the reward
        reward = (profit_reward_weight * positive_profit) - ((1 - profit_reward_weight) * mean_drawdown)

        return reward


def reward_function_sortino(history: History, win_size: int = 144) -> float:
    """
    Reward Function

    Args:
        History
        Window Size

    Returns:
        Sortino ratio
    """
    # adjust based on some baseline
    risk_free_rate = 0.0

    # pull data from history
    data_close = np.array(history['data_close'][-win_size:], dtype=np.float64)
    pos = np.array(history['position'][-win_size:], dtype=np.float64)

    # calculate log returns for the rolling window
    log_data = np.log(data_close)
    log_return = np.where(pos[1:] == 1, np.diff(log_data), 0)

    # Calculate average profit in the window
    average_profit = np.mean(np.where(log_return > 0, log_return, 0))

    # Calculate average loss in the window
    average_loss = np.mean(np.where(log_return < risk_free_rate, log_return^2, 0))

    reward = average_profit/(average_loss)^0.5

    return reward
