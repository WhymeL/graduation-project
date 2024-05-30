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

def generate_actions(current_prices, ma15, ma30):
    """
    Generate action logits based on the comparison between current prices and moving averages.
    """
    logits = np.zeros_like(current_prices, dtype=float)
    
    # Buy if the price is less than MA15 and above MA30
    buy_signals = (current_prices < ma15) & (current_prices > ma30)
    sell_signals = (current_prices > ma15) & (current_prices < ma30)

    # Adjust logits based on the strength of the signal
    # Strength is measured as the percentage difference from the MA
    buy_strength = (ma15 - current_prices) / current_prices
    sell_strength = (current_prices - ma15) / current_prices

    logits[buy_signals] += buy_strength[buy_signals] * 2  # Multiply by a factor to increase influence
    logits[sell_signals] -= sell_strength[sell_signals] * 2  # Multiply by a factor to increase influence

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
        ma15 = np.array([self.data[key]['Close'].rolling(window=15, min_periods=1).mean().iloc[self.current_step] for key in self.data])
        ma30 = np.array([self.data[key]['Close'].rolling(window=30, min_periods=1).mean().iloc[self.current_step] for key in self.data])

        # Combine all features into a single observation array for the current step
        current_observation = np.column_stack((current_prices, current_opens, current_volumes, ma15, ma30))
        current_observation = current_observation.flatten()
        self.dataset['observations'].append(current_observation)

        # action loading
        if self.current_step == 0:
            action = np.array([0.25, 0.25, 0.25, 0.25])
            initial_investment_per_stock = self.current_capital * action
            self.stock_quantities = initial_investment_per_stock / current_opens
            self.dataset['actions'].append(action)
        else:
            # 执行均值回归策略制定动作，并将动作放入dataset['actions']中
            logits = generate_actions(current_prices, ma15, ma30)
            action = np.array(softmax(logits))
            self.stock_quantities = (self.current_capital * action) / current_opens
            self.dataset['actions'].append(action)

        # Calculate total asset after action
        portfolio_value_end_of_day = np.dot(self.stock_quantities, current_prices)
        portfolio_value_start_of_day = np.dot(self.stock_quantities, current_opens)
        daily_return = portfolio_value_end_of_day - portfolio_value_start_of_day

        # reward = daily_return / self.current_capital

        self.current_capital += daily_return

        # 根据总资产计算夏普比率，作为rewards
        # returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        # if len(returns) > 1:
        #     risk_free_rate = 0.01  # 无风险利率
        #     excess_returns = returns - risk_free_rate / 252
        #     sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
        #     reward = sharpe_ratio
        # else:
        #     reward = 0

        self.dataset['rewards'].append(daily_return)
        # self.dataset['rewards'].append(reward)

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
            

def transform_data(dataset):
    data = dataset.copy()
    obs_normalizer = StandardNormalizer()
    act_normalizer = StandardNormalizer()
    rew_normalizer = StandardNormalizer()
    next_obs_normalizer = StandardNormalizer()
    act_normalizer.fit(np.array(data['actions']))
    obs_normalizer.fit(np.array(data['observations']))
    next_obs_normalizer.fit(np.array(data['next_observations']))
    rew_normalizer.fit(np.array(data['rewards']))
    data['observations'] = obs_normalizer.transform(np.array(data['observations']))
    data['actions'] = act_normalizer.transform(np.array(data['actions']))
    data['next_observations'] = next_obs_normalizer.transform(np.array(data['next_observations']))
    data['rewards'] = rew_normalizer.transform(np.array(data['rewards']))
    return data


if __name__ == '__main__':
    file_path = "D:/NJUPT/coding/final_project/mopo-master/historical_data.xlsx"
    stocks_data = load_data(file_path)
    dataset_creator = OfflineDataset(stocks_data)
    while dataset_creator.current_step < dataset_creator.max_step:
        dataset_creator.step()
    final_capital = dataset_creator.current_capital
    print(np.sum(dataset_creator.dataset['rewards']))
    # plt.subplot(1, 2, 1)
    # plt.plot(dataset_creator.dataset['rewards'])
    # plt.title("Daily return of offline dataset")
    # plt.subplot(1, 2, 2)
    # plt.plot(dataset_creator.dataset['rewards'][777: 888])
    # plt.title("Part of the rewards")
    # plt.show()
    
    
    # data = transform_data(dataset_creator.dataset)
    # print(np.isnan(data['observations']))

    # print(np.sum(data['rewards']))
    # plt.plot(data['rewards'])
    # plt.show()    