import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.normalizer import StandardNormalizer

def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    stocks_data = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    return stocks_data

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def generate_actions(current_prices, past_prices):
    """
    Generate action logits based on the momentum strategy.
    Buy stocks with the highest recent price performance.
    """
    # Calculate momentum as the percentage change from past prices to current prices
    momentum = (current_prices - past_prices) / past_prices

    # Initialize logits based on momentum
    logits = np.zeros_like(current_prices, dtype=float)
    logits = momentum  # Higher momentum gets higher logits

    return logits

class OfflineDataset:
    def __init__(self, data):
        self.data = data
        self.num_stocks = len(data)
        self.dates = data[list(data.keys())[0]].index
        self.current_step = 0
        self.max_step = len(self.dates) - 1
        self.initial_capital = 50000
        self.current_capital = self.initial_capital
        # 初始化数据集结构
        self.dataset = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}
    
    def step(self):
        # Fetch current market data for each stock
        current_prices = np.array([self.data[key].loc[self.dates[self.current_step], 'Close'] for key in self.data])
        current_opens = np.array([self.data[key].loc[self.dates[self.current_step], 'Open'] for key in self.data])
        current_volumes = np.array([self.data[key].loc[self.dates[self.current_step], 'Volume'] for key in self.data])
        # Use past prices for momentum calculation (e.g., use prices 5 days ago)
        lookback_period = 5
        if self.current_step >= lookback_period:
            past_prices = np.array([self.data[key].loc[self.dates[self.current_step - lookback_period], 'Close'] for key in self.data])
        else:
            past_prices = current_prices  # If not enough data, use current prices to avoid errors

        # Combine all features into a single observation array for the current step
        ma15 = np.array([self.data[key]['Close'].rolling(window=15, min_periods=1).mean().iloc[self.current_step] for key in self.data])
        ma30 = np.array([self.data[key]['Close'].rolling(window=30, min_periods=1).mean().iloc[self.current_step] for key in self.data])
        current_observation = np.column_stack((current_prices, current_opens, current_volumes, ma15, ma30))
        current_observation = current_observation.flatten()
        self.dataset['observations'].append(current_observation)

        # Action loading
        if self.current_step == 0:
            action = np.array([0.25, 0.25, 0.25, 0.25])
            initial_investment_per_stock = self.current_capital * action
            self.stock_quantities = initial_investment_per_stock / current_opens
            self.dataset['actions'].append(action)
        else:
            # Use momentum strategy to generate actions and append to dataset
            logits = generate_actions(current_prices, past_prices)
            action = np.array(softmax(logits))
            self.stock_quantities = (self.current_capital * action) / current_opens
            self.dataset['actions'].append(action)

        # Calculate total asset after action
        portfolio_value_end_of_day = np.dot(self.stock_quantities, current_prices)
        portfolio_value_start_of_day = np.dot(self.stock_quantities, current_opens)
        daily_return = portfolio_value_end_of_day - portfolio_value_start_of_day

        # option 1
        reward = daily_return / self.current_capital

        self.current_capital += daily_return

        # self.dataset['rewards'].append(daily_return)
        self.dataset['rewards'].append(reward)
        self.current_step += 1

        next_obs = self._next_observations()
        self.dataset['next_observations'].append(np.array(next_obs))

        done = self.current_step >= self.max_step - 1
        self.dataset['terminals'].append(done)
    
    def _next_observations(self):
        if self.current_step < self.max_step - 1:
            prices = np.array([self.data[key].loc[self.dates[self.current_step], 'Close'] for key in self.data])
            opens = np.array([self.data[key].loc[self.dates[self.current_step], 'Open'] for key in self.data])
            volumes = np.array([self.data[key].loc[self.dates[self.current_step], 'Volume'] for key in self.data])
            ma15 = np.array([self.data[key]['Close'].rolling(window=15, min_periods=1).mean().iloc[self.current_step] for key in self.data])
            ma30 = np.array([self.data[key]['Close'].rolling(window=30, min_periods=1).mean().iloc[self.current_step] for key in self.data])

            # Combine all features into a single observation array for the current step
            next_observation = np.column_stack((prices, opens, volumes, ma15, ma30))
            next_observation = next_observation.flatten()
        else:
            prices = np.array([self.data[key].loc[self.dates[self.max_step], 'Close'] for key in self.data])
            opens = np.array([self.data[key].loc[self.dates[self.max_step], 'Open'] for key in self.data])
            volumes = np.array([self.data[key].loc[self.dates[self.max_step], 'Volume'] for key in self.data])
            ma15 = np.array([self.data[key]['Close'].rolling(window=15, min_periods=1).mean().iloc[self.max_step] for key in self.data])
            ma30 = np.array([self.data[key]['Close'].rolling(window=30, min_periods=1).mean().iloc[self.max_step] for key in self.data])

            # Combine all features into a single observation array for the current step
            next_observation = np.column_stack((prices, opens, volumes, ma15, ma30))
            next_observation = next_observation.flatten() 
        return next_observation


if __name__ == '__main__':
    file_path = "D:/NJUPT/coding/final_project/mopo-master/historical_data.xlsx"
    stocks_data = load_data(file_path)
    dataset_creator = OfflineDataset(stocks_data)
    while dataset_creator.current_step < dataset_creator.max_step:
        dataset_creator.step()
    final_capital = dataset_creator.current_capital
    print(final_capital)
    print(np.sum(dataset_creator.dataset['rewards']))
