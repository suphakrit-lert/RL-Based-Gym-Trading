# RL Based Gym Trading

`AnyTrading` is a collection of [OpenAI Gym](https://github.com/openai/gym) environments for reinforcement learning-based trading algorithms.

## Project Description
Our project aims to develop a novel trading strategy that integrates reinforcement learning (RL) techniques, specifically A2C (Advantage Actor-Critic) and PPO (Proximal Policy Optimization), with financial factors to maximize returns in trading markets. Leveraging the Gym-anytrading framework under the OpenAI Gym Environment, we will build and test RL-based trading algorithms within realistic trading environments. Additionally, we will incorporate financial factors such as economic indicators, market sentiment analysis, and fundamental analysis to enhance the effectiveness of the trading strategy. By combining RL with financial domain knowledge, we aim to create a robust and adaptive trading system capable of navigating dynamic market conditions and achieving superior returns. 

## Resources
We will utilize the Gym-anytrading framework along with the quantstats package for building and  testing our RL-based trading algorithms. Additionally, we will gather historical financial data from  reputable sources such as Bloomberg, Yahoo Finance, and Kaggle datasets. This data will include  historical price data, economic indicators, company financial reports, and other relevant financial  factors. Version control will be managed using GitHub to ensure effective collaboration and code  management. 

## Installation

_Step to setup_

1. Create a virtual environment
   ```sh
   py -3.12 -m venv .venv
   ```

2. Enter the newly created virtual environment
   ```sh
   .venv\Scripts\activate
   ```

3. Install all the dependencies in the requirements.txt
   ```sh
   py -m pip install -r requirements.txt
   ```
