import numpy as np


class StaticFns:
    @staticmethod
    # def termination_fn(obs, next_obs, max_steps, current_step, initial_balance, balance):
    #     """
    #     Determine whether the simulation should terminate.

    #     Args:
    #     obs (np.array): The current observation.
    #     next_obs (np.array): The next observation after taking an action.
    #     max_steps (int): The maximum number of steps in the simulation.
    #     current_step (int): The current step in the simulation.
    #     initial_balance (float): The initial balance at the start of the simulation.
    #     balance (float): The current balance.

    #     Returns:
    #     bool: True if the simulation should terminate, False otherwise.
    #     """
    #     # Check if the current step has exceeded the maximum allowed steps
    #     if current_step >= max_steps:
    #         return True
        
    #     # Check if the portfolio value has dropped below 50% of the initial balance
    #     if balance < initial_balance * 0.5:
    #         return True
        
    #     # Check if the next observations are finite and within expected ranges
    #     # For example, ensuring no stock prices or volumes are negative
    #     if not np.isfinite(next_obs).all():
    #         return True
        
    #     # Here, you might add more conditions based on other financial indicators or risk measures
        
    #     # If none of the termination conditions are met, continue the simulation
    #     return False
    def termination_fn(obs, act, next_obs):
        done = np.array([False]).repeat(len(obs))
        return done