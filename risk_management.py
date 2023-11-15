# Author: James Mejia
# Date: 11/01/2023
# Description:

class RiskData:

    def __init__(self, initial_balance):
        self.__in_market = False
        self.__flat_market_days = 0
        self.__initial_value = initial_balance
        self.__initial_value_percent_change = 0.0
        self.__current_value = initial_balance
        self.__current_value_percent_change = 0.00
        self.__high_value = 0
        self.__high_value_percent_change = 0
        self.__stop_loss = 80_000.00
        self.__buy_line = initial_balance

    # GETTERS
    def get_in_market(self):
        return self.__in_market

    def get_flat_market_days(self):
        return self.__flat_market_days

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

    def get_buy_line(self):
        return self.__buy_line

    # SETTERS
    def set_in_market(self, in_market: bool):
        self.__in_market = in_market

    def set_flat_market_days(self, flat_market_days: int):
        self.__flat_market_days = flat_market_days

    def set_initial_value(self, initial_value):
        self.__initial_value = initial_value

    def set_initial_value_percent_change(self, initial_value_percent_change):
        self.__initial_value_percent_change = initial_value_percent_change

    def set_current_value(self, current_value):
        self.__current_value = current_value

    def set_current_value_percent_change(self, current_value_percent_change):
        self.__current_value_percent_change = current_value_percent_change

    def set_high_value(self, high_value):
        self.__high_value = high_value

    def set_high_value_percent_change(self, value_high_percent_change):
        self.__high_value_percent_change = value_high_percent_change

    def update_risk_data(self, portfolio_balance: int):
        # Set Percent Change if just got in market
        if self.__in_market is False:
            self.__initial_value = portfolio_balance
            self.__initial_value_percent_change = 0.0
            self.__current_value = portfolio_balance
            self.__current_value_percent_change = 0
            self.__high_value = portfolio_balance
            self.__high_value_percent_change = 0
            self.__in_market = True

        else:
            # Set Percent Changes and set new current value and high value (Increase in value)
            if self.__current_value < portfolio_balance:
                self.__current_value_percent_change = (portfolio_balance / self.__initial_value) * 0.01
                self.__current_value = portfolio_balance
                self.__high_value = portfolio_balance
                self.__high_value_percent_change = 0.0
            else:
                # Set Percent Changes (Decrease in value)
                self.__current_value_percent_change = -1 * (1 - (portfolio_balance / self.__initial_value))
                self.__current_value = portfolio_balance
                self.__high_value_percent_change = -1 * (1 - (portfolio_balance / self.__high_value))
                # print("Percent Change: ", self.__current_value_percent_change,
                #       " High Value % Change: ", self.__high_value_percent_change)

        # print("Days ", info["step"], " Days In Market: ", self.__days_in_market)
        # print("Current Value: ", self.__current_value)

    def reset_risk_values(self):
        self.__in_market = False
        self.__current_value = 0
        self.__flat_market_days = 0

    def run_risk_analysis(self, portfolio_balance: int) -> dict:

        # MAX PORTFOLIO LOSS
        if portfolio_balance < self.__stop_loss:
            # print("Max Loss Triggered")
            return {"too_much_risk": True,
                    "risk_reward": -100}

        # BUY LINE LOSS
        if portfolio_balance < self.__buy_line:
            # print("Buy Line Loss")
            return {"too_much_risk": True,
                    "risk_reward": -100}

        # TEMPORAL LOSS
        if 0.01 >= self.__current_value_percent_change >= -0.01 and 3 >= self.__flat_market_days > 0:
            # print("Temporal Stop Loss")
            too_much_risk = True
            risk_value = self.__flat_market_days
            risk_value = abs(risk_value) ** 4
            if risk_value > 100:
                risk_value = -100
            else:
                risk_value = -risk_value
                self.set_flat_market_days += 1

            return {"too_much_risk": too_much_risk,
                    "risk_reward": risk_value}

        # MAX PERCENT LOSS
        if self.__high_value_percent_change < -.02:
            # print("LOSS > 2%, issues sell request")
            too_much_risk = True
            risk_value = 2 + self.__high_value_percent_change
            risk_value = abs(risk_value) ** 4
            if risk_value > 100:
                risk_value = -100
            else:
                risk_value = -risk_value

            return {"too_much_risk": too_much_risk,
                    "risk_reward": risk_value}

        return {"too_much_risk": False,
                "risk_reward": 10}


def __max_loss(info: dict, risk_data: RiskData) -> dict:
    # HARD SELL, HIT BOTTOM
    if info["portfolio_balance"] < risk_data.get_stop_loss() and info["signal"] == 1:
        # print("Max Loss Triggered")
        return {"too_much_risk": True,
                "risk_reward": -100}

    return {"too_much_risk": False,
            "risk_reward": 10}


def __buy_line_loss(info: dict, risk_data: RiskData) -> dict:
    # HARD SELL, HIT BUY LINE
    if info["portfolio_balance"] < risk_data.get_buy_line() and info["signal"] == 1:
        # print("Buy Line Loss")
        return {"too_much_risk": True,
                "risk_reward": -100}

    return {"too_much_risk": False,
            "risk_reward": 10}


def __temporal_loss(risk_data: RiskData) -> dict:
    # SELL IF TIME IN MARKET YIELDS LITTLE TO NO GAIN - Flat Market is < 1% for 3 days or more
    if 0.01 >= risk_data.get_current_value_percent_change() >= -0.01 and 3 >= risk_data.get_flat_market_days() > 0:
        # print("Temporal Stop Loss")
        too_much_risk = True
        risk_value = risk_data.get_flat_market_days()
        risk_value = abs(risk_value) ** 4
        if risk_value > 100:
            risk_value = -100
            risk_data.reset_risk_values()
        else:
            risk_value = -risk_value
            risk_data.set_flat_market_days(risk_data.get_flat_market_days() + 1)
            print("Flat Days:", risk_data.get_flat_market_days())

        return {"too_much_risk": too_much_risk,
                "risk_reward": risk_value}

    return {"too_much_risk": False,
            "risk_reward": 0}


def __risk_percent(info: dict, risk_data: RiskData) -> dict:
    if risk_data.get_high_value_percent_change() < -.02 and info["signal"] == 1:
        # print("LOSS > 2%, issues sell request")
        too_much_risk = True
        risk_value = 2 + risk_data.get_high_value_percent_change()
        risk_value = abs(risk_value) ** 4
        if risk_value > 100:
            risk_value = -100
        else:
            risk_value = -risk_value

        risk_data.reset_risk_values()
        return {"too_much_risk": too_much_risk,
                "risk_reward": risk_value}

    return {"too_much_risk": False,
            "risk_reward": 10}
