import numpy as np
import gym
from gym import spaces



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

'''
设置初始投资资金initial capital
在每个时间步记录采取动作后的即时收益作为奖励
在采取动作后计算当前总资产current capital
当current capital超过初始资金的30%时，将current capital设置为初始资金量
当current capital小于等于0或走完整个交易周期，terminal设置为True
'''
class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.num_stocks = np.array(data['actions']).shape[1]
        self.dates = np.array(data['actions']).shape[0]
        self.current_step = 0
        self.max_steps = self.dates - 1
        self.initial_capital = 50000
        self.current_capital = self.initial_capital

        # Observation and action space
        num_features = 5  # close, open, volumes, MA15, MA30
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_stocks * num_features,), dtype=np.float32)

    def step(self, action):
        action = softmax(action)

        # Fetch current market data for each stock
        current_obs = self._next_observation()
        current_close_prices = current_obs[::5]
        current_open_prices = current_obs[1::5]

        # Calculate reward: simple version could just be the change in portfolio value
        if self.current_step == 0:
            initial_investment_per_stock = self.current_capital * action
            self.stock_quantities = initial_investment_per_stock  / current_open_prices
        else:
            self.stock_quantities = (self.current_capital * action) / current_open_prices
        
        portfolio_value_end_of_day = np.dot(self.stock_quantities, current_close_prices)
        portfolio_value_start_of_day = np.dot(self.stock_quantities, current_open_prices)
        daily_return = portfolio_value_end_of_day - portfolio_value_start_of_day

        # test
        # print("daily_return: ", daily_return)

        reward = daily_return / self.current_capital
        self.current_capital += daily_return
        # if self.current_capital >= self.initial_capital * (1 + 0.3):
        #     self.current_capital = self.initial_capital
        # print("current_capital: ", self.current_capital)
        
        
        # Increment the step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_observation = self._next_observation() if not done else np.zeros_like(current_obs)
        
        # Return step information
        return next_observation, reward, done, {}

    def reset(self):
        # Reset environment to initial state
        self.current_step = 0
        self.current_capital = self.initial_capital
        return self._next_observation()

    def _next_observation(self):
        return self.data['observations'][self.current_step]

    def render(self, mode='human'):
        # Optional: Implement rendering for visualization
        pass

    def close(self):
        # Optional: Implement cleanup
        pass
