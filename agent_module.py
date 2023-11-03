from stable_baselines3 import PPO
import torch
from datetime import datetime

import risk_management
from risk_management import *
import os

RENDER_DIR = 'render_logs'



class PPOAgentModule:
    def __init__(self, env, model_path=None):
        self.env = env
        self.model_path = model_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if model_path:
            self.model = PPO.load(model_path)
        else:
            self.model = PPO('MlpPolicy',
                             env=self.env,
                             verbose=0,
                             gamma=0.7,
                             n_steps=200,
                             ent_coef=0.01,
                             learning_rate=0.0003,
                             clip_range=0.1,
                             batch_size=10,
                             device=self.device)

    def train(self, total_timesteps):
        """
        Trains the agent for the given number of timesteps.
        Args:
            total_timesteps (int): Number of timesteps to train the agent for.

        Returns:
            None
        """
        print("Using device:", self.device)
        print("Training.")
        self.model.learn(total_timesteps=total_timesteps)

        # Save model
        model_dir = './models/'
        os.makedirs(model_dir, exist_ok=True)
        curr_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        self.model.save(f"models/{curr_datetime}_ppo_trading_agent")
        print("Training complete.")
        print("Model saved at path:",
              f"models/{curr_datetime}_ppo_trading_agent")

    def test(self, test_env, testing_df):
        """
        Tests the agent on the given testing data.
        Args:
            test_env (TradingEnv): Trading environment to test the agent on.
            testing_df (pd.DataFrame): Testing data.
            render_dir (str): Directory to save the render in.

        Returns:
            None
        """
        print("Using device:", self.device)
        observation, info = test_env.reset()
        risk_data = RiskData()
        print("Testing model on testing data.")
        for _ in range(len(testing_df)):
            position_index, _state = self.model.predict(observation=observation)

            if position_index == 1:
                # Run Risk Analysis After Prediction
                risk_data.update_risk_data(info)
                if risk_management.run_risk_analysis(info, risk_data):
                    position_index = 0

            observation, reward, done, truncated, info = test_env.step(position_index)

            if done or truncated:
                break

        print("Portfolio Valuation: ", info["portfolio_valuation"])

        # Save render
        if not os.path.exists(RENDER_DIR):
            os.makedirs(RENDER_DIR)
        test_env.get_wrapper_attr('save_for_render')(dir=RENDER_DIR)
        print(f"Test finished. Render saved in {RENDER_DIR}")
