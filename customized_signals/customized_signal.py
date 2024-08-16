# Gym stuff
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import RecurrentPPO

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import argparse
import yfinance as yf
import quantstats as qs
from finta import TA

import sys

SEED = 5756
# Usage: py customized_signal.py --stock_name MCD --train_window_size 10 --train_start 2023-01-01 --train_end 2023-09-10 --test_window_size 10 --learn_iteration 100 --model_name A2C

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CustomizedSignal",
        description="Train RL for Trading",
        epilog="Example: py customized_signal.py --name AMD --window-size 10",
    )
    parser.add_argument(
        "--stock_name",
        nargs=1,
        help="Name of the Stock",
    )
    parser.add_argument(
        "--train_window_size",
        nargs=1,
        help="Train Window Size",
    )
    parser.add_argument(
        "--train_start",
        nargs=1,
        help="Train Start",
    )
    parser.add_argument(
        "--train_end",
        nargs=1,
        help="Train End",
    )
    parser.add_argument(
        "--test_window_size",
        nargs=1,
        help="Test Window Size",
    )
    parser.add_argument(
        "--learn_iteration",
        nargs=1,
        help="Learn Iteration",
    )
    parser.add_argument(
        "--model_name",
        nargs=1,
        help="Model Use",
    )

    args = parser.parse_args()

    # Specify parameters
    stock_name = args.stock_name[0]

    # Window size for training
    train_window_size = int(args.train_window_size[0])
    train_start = args.train_start[0]
    train_end = args.train_end[0]

    test_window_size = int(args.test_window_size[0])
    learn_iteration = int(args.learn_iteration[0])

    # Model parameters
    model_name = args.model_name[0]   

    print(f'Stock Name: {stock_name}\nTrain Window Size: {train_window_size}')
    print(f'Train Start: {train_start}\nTrain End: {train_end}')
    print(f'Test Window Size: {test_window_size}\nLearn Iteration: {learn_iteration}')
    print(f'Model Name: {model_name}')

    df = yf.Ticker(args.stock_name[0])
    df = df.history(period="max")
    df = df.loc['2020-01-01':, ['Open', 'High', 'Low', 'Close', 'Volume']]
    TRAIN_ENV_FRAME_BOUND = (train_window_size, df[train_start:train_end].shape[0])
    TEST_ENV_FRAME_BOUND = (df[train_start:train_end].shape[0] + test_window_size, df.shape[0])

    df['SMA'] = TA.SMA(df, 10)
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)
    df.fillna(0, inplace=True)

    def add_signals(env):
        start = env.frame_bound[0] - env.window_size
        end = env.frame_bound[1]
        prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
        signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
        return prices, signal_features
    
    class MyCustomEnv(StocksEnv):
        _process_data = add_signals

    env2 = MyCustomEnv(df=df, frame_bound=TRAIN_ENV_FRAME_BOUND, window_size=train_window_size)
    env_maker = lambda: env2
    env = DummyVecEnv([env_maker])

    if model_name == 'A2C':
        model = A2C('MlpPolicy', env, verbose=0, seed=SEED) 
        model.learn(total_timesteps=learn_iteration)
    elif model_name == 'RecurrentPPO':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, seed=SEED)
        model.learn(total_timesteps=learn_iteration)
    elif model_name == 'PPO':
        model = PPO('MlpPolicy', env, verbose=0, seed=SEED) 
        model.learn(total_timesteps=learn_iteration)
    elif model_name == 'DQN':
        model = DQN('MlpPolicy', env, verbose=0, seed=SEED) 
        model.learn(total_timesteps=learn_iteration)

    env = MyCustomEnv(df=df, frame_bound=TEST_ENV_FRAME_BOUND, window_size=test_window_size)
    observation, info = env.reset()
    while True: 
        observation = observation[np.newaxis, ...]
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("info", info)
            break
    
    total_profit = round(info['total_profit'], 2)

    # Report output
    trade_strategy_output = f".\\reports\\{stock_name}_{model_name}_PROFIT{total_profit}_WINSIZE{train_window_size}_ITER{learn_iteration}_trade_strategy.png"
    report_output = f".\\reports\\{stock_name}_{model_name}_PROFIT{total_profit}_WINSIZE{train_window_size}_ITER{learn_iteration}_reports.html"

    desc = f'STOCK_NAME: {stock_name}\nTRAIN_WINDOW_SIZE: {train_window_size}\nTRAIN_START: {train_start}\nTRAIN_END: {train_end}\nTEST_WINDOW_SIZE: {test_window_size}\nLEARN_ITERATIONS: {learn_iteration}\nSMA_PERIOD: 10'
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.text(0, env.prices.max() - 5, desc)
    plt.savefig(trade_strategy_output)

    qs.extend_pandas()

    net_worth = pd.Series(env.unwrapped.history['total_profit'], index=df.index[TEST_ENV_FRAME_BOUND[0] + 1:TEST_ENV_FRAME_BOUND[1]])
    returns = net_worth.pct_change().iloc[1:]

    qs.reports.html(returns, output=report_output)