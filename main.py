from reward_function import reward_function_drawdown
from agent_module import PPOAgentModule
from data_processor import DataProcessor
import gymnasium as gym
import pandas as pd


def main():
    data_processor = DataProcessor()
    symbol = 'TQQQ'
    start_date = '2015-01-01'
    stop_date = '2023-10-01'
    tqqq = data_processor.download_data_df_from_yf(symbol,
                                                   start_date,
                                                   stop_date)
    tqqq_preprocessed = data_processor.preprocess_data(tqqq, 50)

    # Format table date proper format and name
    tqqq_preprocessed.dropna(inplace=True)  # Clean again !

    # Format to gym-trader-env format
    df = pd.DataFrame([], [])
    for column in tqqq_preprocessed.columns:
        df[str(column).lower()] = pd.DataFrame(tqqq_preprocessed[column].values, columns=[column])
    df["date"] = pd.DataFrame(tqqq_preprocessed["Close"].index, columns=["Date"])
    df.head()

    # Setup training data
    training_df = df[df["date"] <= "2022-12-31"].copy()
    training_df.dropna(inplace=True)
    training_df.head()

    # Setup testing data
    testing_df = df[df["date"] > "2022-12-31"].copy()
    testing_df.dropna(inplace=True)
    testing_df.head()

    def trainer():
        #  load training environment
        training_env = gym.make("TradingEnv",
                                df=training_df,
                                positions=[0, 1],
                                initial_position=1,
                                portfolio_initial_value=1000,
                                reward_function=reward_function_drawdown)
        # Train model
        agent = PPOAgentModule(training_env)
        agent.train(10000)

    def tester():
        # Load testing environment
        testing_env = gym.make("TradingEnv",
                               df=testing_df,
                               positions=[0, 1],
                               initial_position=1,
                               portfolio_initial_value=1000,
                               reward_function=reward_function_drawdown)
    
        # Load model and agent
        agent = PPOAgentModule(testing_env, model_path="models/20231101170410_ppo_trading_agent.zip")
        print(agent)
        agent.test(testing_env, testing_df)
    
    while True:
        print("1. Train")
        print("2. Test")
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a valid choice.")
            continue
        print("\n\n")
        if choice == 1:
            trainer()
            break
        elif choice == 2:
            tester()
            break
        print("\n\n")


if __name__ == '__main__':
    main()
