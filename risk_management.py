# Author: James Mejia
# Date: 11/01/2023
# Description: Risk management consists of 5 triggers

import numpy as np
from gym_trading_env.utils.history import History


class RiskData:

    def __init__(self):
        self.__in_market = False
        self.__days_in_market = 0
        self.__initial_value = 1000
        self.__initial_value_percent_change = 0.0
        self.__current_value = 1000
        self.__current_value_percent_change = 0.00
        self.__percent_change = 0.00
        self.__high_value = 0
        self.__high_value_percent_change = 0
        self.__stop_loss = 800
        self.__buy_loss = 1000

    # GETTERS
    def get_in_market(self):
        return self.__in_market

    def get_days_in_market(self):
        return self.__days_in_market

    def get_initial_value(self):
        return self.__initial_value

    def get_initial_value_percent_change(self):
        return self.__initial_value_percent_change

    def get_current_value(self):
        return self.__current_value

    def get_current_value_percent_change(self):
        return self.__current_value_percent_change

    def get_high_value(self):
        return self.__high_value

    def get_high_value_percent_change(self):
        return self.__high_value_percent_change

    def get_stop_loss(self):
        return self.__stop_loss

    def get_buy_loss(self):
        return self.__buy_loss

    # SETTERS
    def set_in_market(self, in_market : bool):
        self.__in_market = in_market

    def set_days_in_market(self, days_in_market: int):
        self.__days_in_market = days_in_market

    def set_initial_value(self, initial_value):
        self.__initial_value = initial_value

    def set_initial_value_percent_change(self, initial_value_percent_change):
        self.__initial_value_percent_change = initial_value_percent_change

    def set_current_value(self, current_value):
        self.__current_value = current_value

    def set_current_value_percent_change(self, current_value_percent_change):
        self.__current_value_percent_change = current_value_percent_change

    def set_high_value(self, high_value):
        self.__high_value= high_value

    def set_high_value_percent_change(self, value_high_percent_change):
        self.__high_value_percent_change = value_high_percent_change

    def update_risk_data(self, info: dict):
        # Set Percent Change if just got in market
        if self.__in_market is False:
            self.__initial_value = info["portfolio_valuation"]
            self.__initial_value_percent_change = 0.0
            self.__current_value = info["portfolio_valuation"]
            self.__current_value_percent_change = 0
            self.__high_value = info["portfolio_valuation"]
            self.__high_value_percent_change = 0
            self.__in_market = True
            self.__days_in_market = 1

        else:
            # Set Percent Changes and set new current value and high value (Increase in value)
            if self.__current_value < info["portfolio_valuation"]:
                self.__current_value_percent_change = (info["portfolio_valuation"] / self.__initial_value) * 0.01
                self.__current_value = info["portfolio_valuation"]
                self.__high_value = info["portfolio_valuation"]
                self.__high_value_percent_change = 0.0
            else:
                # Set Percent Changes (Decrease in value)
                self.__current_value_percent_change = -1 * (1 - (info["portfolio_valuation"] / self.__initial_value))
                self.__current_value = info["portfolio_valuation"]
                self.__high_value_percent_change = -1 * (1 - (info["portfolio_valuation"] / self.__high_value))
                # print("Percent Change: ", self.__current_value_percent_change,
                #       " High Value % Change: ", self.__high_value_percent_change)

            self.__days_in_market += 1

        # print("Days ", info["step"], " Days In Market: ", self.__days_in_market)
        # print("Current Value: ", self.__current_value)

    def reset_risk_values(self):
        self.__in_market = False
        self.__current_value = 0
        self.__days_in_market = 0


def run_risk_analysis(info: dict, risk_data: RiskData):

    if __max_loss(info, risk_data) is True:
        return True

    if __buy_line_loss(info, risk_data) is True:
        return True

    if __temperal_loss(info, risk_data) is True:
        return True

    if __risk_reward(info, risk_data) is True:
        return True

    return False

def __risk_reward(info: dict, risk_data: RiskData) -> bool:
    if risk_data.get_high_value_percent_change() < -.02 and info["position"] == 1:
        # print("LOSS > 2%, issues sell request")
        risk_data.reset_risk_values()
        return True

def __temperal_loss(info: dict, risk_data: RiskData) -> bool:
    # SELL IF TIME IN MARKET YEILDS LITTLE TO NO GAIN - Flat Market is < 1% for 3 days or more
    if risk_data.get_days_in_market() >= 3:
        if risk_data.get_current_value_percent_change() < 0.01:
            # print("Temperal Stop Loss")
            risk_data.reset_risk_values()
            return True

def __buy_line_loss(info: dict, risk_data: RiskData) -> bool:
    # HARD SELL, HIT BUY LINE
    if info["portfolio_valuation"] < risk_data.get_buy_loss() and info["position"] == 1:
        # print("Buy Line Loss")
        risk_data.reset_risk_values()
        return True

def __max_loss(info: dict, risk_data: RiskData) -> bool:
    # HARD SELL, HIT BOTTOM
    if info["portfolio_valuation"] < risk_data.get_stop_loss() and info["position"] == 1:
        # print("Max Loss Triggered")
        risk_data.reset_risk_values()
        return True
