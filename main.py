from agent_module import PPOAgentModule
from data_processor import DataProcessor
# from gymnasium import gym
from asset_trading_env import AssetTradingEnv


def main():
    dp = DataProcessor()
    symbol = 'TQQQ'

    def trainer():
        start_date = '2010-01-01'
        stop_date = '2020-01-01'
        tqqq = dp.download_data_df_from_yf(
            symbol, start_date, stop_date)
        training_df = dp.preprocess_data(tqqq)
        training_df.fillna(0, inplace=True)

        #  load training environment
        training_env = AssetTradingEnv(data_df=training_df)
        # training_env = gym.make("TradingEnv",
        #                         df=training_df,
        #                         positions=[-1, 0, 1],
        #                         initial_position=1,
        #                         portfolio_initial_value=100000,
        #                         reward_function=drawdown)
        # Train model
        agent = PPOAgentModule(training_env)
        agent.train(10000)

    def tester():

        start_date = '2020-01-02'
        stop_date = '2023-11-30'
        tqqq = dp.download_data_df_from_yf(symbol, start_date, stop_date)
        testing_df = dp.preprocess_data(tqqq)
        testing_df.fillna(0, inplace=True)

        # Load testing environment
        testing_env = AssetTradingEnv(data_df=testing_df)
        # testing_env = gym.make("TradingEnv",
        #                        df=testing_df,
        #                        positions=[-1, 0, 1],
        #                        initial_position=1,
        #                        portfolio_initial_value=100000,
        #                        reward_function=drawdown)

        # Load model and agent
        print("Testing model on testing data.")
        for test in range(20):
            agent = PPOAgentModule(
                testing_env,
                model_path="models/20231113121730_ppo_trading_agent.zip")
            agent.test(testing_env, testing_df)

    while True:
        print("1. Train")
        print("2. Test")
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a valid choice.")
            continue
        print("\n")
        if choice == 1:
            trainer()
            break
        elif choice == 2:
            tester()
            break
        print("\n")


if __name__ == '__main__':
    main()
