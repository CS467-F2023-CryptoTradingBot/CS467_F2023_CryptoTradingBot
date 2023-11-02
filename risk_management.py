# Author: James Mejia
# Date: 11/01/2023
# Description: Risk management consists of 5 triggers

import numpy as np
from gym_trading_env.utils.history import History


class RiskData:

    def __init__(self):
        self.__high = 1000
        self.__value_change = 0
        self.__percent_change = 0
        self.__stop_loss = 800
        self.__buy_loss = 1000

    def get_high(self):
        return self.__high

    def get_value_change(self):
        return self.__value_change

    def get_percent_change(self):
        return self.__percent_change

    def get_stop_loss(self):
        return self.__stop_loss

    def get_buy_loss(self):
        return self.__buy_loss

    def set_high(self, high):
        self.__high = high

    def set_value_change(self, value_change):
        self.__value_change = value_change

    def set_value_change(self, percent_change):
        self.__percent_change = percent_change

    def update_risk_data(self, info: dict):

        # Set Percent Change
        self.__percent_change = (info["portfolio_valuation"] - self.__high)/self.__high
        print("Percent Change: ", self.__percent_change)

        # Set Value Change
        self.__value_change =  info["portfolio_valuation"] - self.__high
        print("Value Change: ", self.__value_change)

        # Set New High
        if self.__high < info["portfolio_valuation"]:
            self.__high = info["portfolio_valuation"]

        print("High: ", self.__high)


def run_risk_analysis(info: dict, risk_data: RiskData):
    if __max_loss(info, risk_data) is True:
        return True

    if __buy_line_loss(info, risk_data) is True:
        return True

    if __risk_reward(info, risk_data) is True:
        return True

    return False

def __risk_reward(info: dict, risk_data: RiskData) -> bool:
    if (risk_data.get_percent_change()) < .02 and info["position"] == 0:
        print("LOSS > 2%, issues sell request")
        return True

def __buy_line_loss(info: dict, risk_data: RiskData) -> bool:
    # HARD SELL, HIT BUY LINE
    if info["portfolio_valuation"] < risk_data.get_buy_loss() and info["position"] == 0:
        print("Buy Line Loss")
        return True

def __max_loss(info: dict, risk_data: RiskData) -> bool:
    # HARD SELL, HIT BOTTOM
    if info["portfolio_valuation"] < risk_data.get_stop_loss() and info["position"] == 0:
        print("Max Loss Triggered")
        return True