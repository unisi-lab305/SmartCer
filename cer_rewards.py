from cer_models.node_models import Producer, Consumer, Prosumer

from typing import Dict, Tuple, Union

import numpy as np


class Reward:
    """
    Class for reward computation. Implements different reward types.

    Attributes:
        nodes (Dict[str, Union[Producer, Consumer, Prosumer]]): Dictionary of Cer nodes.
        type (str): Type of reward function.
        single_agent (bool): Flag for single-agent or multi-agent reward computation.
        add_penalty (bool): Flag for adding penalty to the reward.
        penalty_coefficient (float): Coefficient for penalty calculation.
        max_ratio (float): Maximum ratio of constraint violation before episode truncation.
    """

    def __init__(self, nodes: Dict[str, Union[Producer, Consumer, Prosumer]], **reward_params: Dict):
        """
        Initializes an instance of the Rewards class.

        Args:
            nodes (Dict[str, Union[Producer, Consumer, Prosumer]]): Dictionary of storage nodes.
            **reward_params (Dict): Dictionary of reward parameters.

        """
        self.nodes = nodes
        try:
            self.type = reward_params['type']
            self.single_agent = reward_params['single_agent']
            self.add_penalty = reward_params['constraints']['penalty']
            self.penalty_coefficient = reward_params['constraints']['coefficient']
            self.max_ratio = reward_params['constraints']['max_ratio']
        except KeyError as e:
            raise KeyError(f'Missing key in reward config file: {e}')

        assert self.type in ['energy', 'cost'], f'Invalid reward type: {self.type}. Available types: [energy, cost]'

    def __repr__(self):
        return f'{self.__class__.__name__} → type: {self.type}, single_agent: {self.single_agent}, ' \
                f'add_penalty: {self.add_penalty}, penalty_coefficient: {self.penalty_coefficient}'

    def compute_penalty(self, time_step: int) -> Dict[str, float] or float:
        """
        Compute penalty for constraint violation. The penalty is normalized by the storage capacity and multiplied by
        the penalty coefficient.
        For single-agent reward, returns a dictionary of penalties for each node. For collective reward, returns the
        sum of all penalties.

        Args:
            time_step (int): Current time step.

        Returns:
            Dict[str, float] or float: Penalty(s) for the current time step.
        """
        penalties = {}
        for node_id, node in self.nodes.items():
            if node.storage:
                normalized_penalty = node.storage.constraint_violation[time_step] / node.storage.capacity
                penalties[node_id] = self.penalty_coefficient * normalized_penalty
        if self.single_agent:
            return penalties
        else:
            return sum(penalties.values())

    def compute_energy_reward(self, time_step: int) -> Dict[str, float] or float:
        """
        Energy reward: reward=sign(trade)*min(0, trade)^2.
        Where 'trade' is computed as: production + discharge - consumption - charge.
        For single-agent reward, returns a dictionary of rewards for each node. For collective reward, returns the sum.
        NOTE: For 'Producer' nodes, the reward is always zero.

        Args:
            time_step (int): Current time step.

        Returns:
            Dict[str, float] or float: Reward(s) for the current time step.
        """
        rewards = {}
        cumulative_reward = 0.0
        if self.single_agent:
            for node_id, node in self.nodes.items():
                if node.storage:
                    trade = node.energy_trade[time_step]
                    rewards[node_id] = np.sign(trade) * min(0, trade)**2
            return rewards
        else:
            for node_id, node in self.nodes.items():
                trade = node.energy_trade[time_step]
                cumulative_reward += np.sign(trade) * min(0, trade)**2
            return cumulative_reward

    def compute_cost_reward(self, tip: float, time_step: int) -> Dict[str, float] or float:
        """
        Cost reward: reward=energy_revenue - energy_expense.
        For single-agent reward, returns a dictionary of rewards for each node.
        For collective reward, returns the sum of all rewards plus the total incentive (tip).

        Args:
            tip (float): Total incentive payment.
            time_step (int): Current time step.

        Returns:
            Dict[str, float] or float: Reward(s) for the current time step.
        """
        rewards = {}
        cumulative_reward = 0.0
        if self.single_agent:
            for node_id, node in self.nodes.items():
                if node.storage:
                    cost = node.energy_revenue[time_step] - node.energy_expense[time_step]
                    rewards[node_id] = cost
            return rewards
        else:
            for node_id, node in self.nodes.items():
                cost = node.energy_revenue[time_step] - node.energy_expense[time_step]
                cumulative_reward += cost
            cumulative_reward += tip
            return cumulative_reward


class RewardScaler:
    """
    Class for scaling and normalizing reward values in the CER environment.

    Attributes:
        mean (float): Running mean of the rewards.
        M2 (float): Accumulator for variance calculations.
        count (int): Count of rewards seen so far.
        epsilon (float): Small value to avoid division by zero in variance calculations.
        clip_min (float): Minimum value for clipping the scaled reward.
        clip_max (float): Maximum value for clipping the scaled reward.
        reset_interval (int, optional): Number of steps after which to reset statistics (None to disable resetting).
        step_counter (int): Counter for the number of updates since the last reset.
    """

    def __init__(self, clip_min=-1.0, clip_max=1.0, reset_interval=None):
        """
        Initializes the RewardScaler.

        Args:
            clip_min (float): Minimum value for clipping the scaled reward. Default is -1.0.
            clip_max (float): Maximum value for clipping the scaled reward. Default is 1.0.
            reset_interval (int, optional): Number of steps after which to reset statistics (None to disable resetting).
        """
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.epsilon = 1e-8
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.reset_interval = reset_interval
        self.step_counter = 0

    def update(self, reward):
        """
        Updates the running mean and variance with a new reward.

        Args:
            reward (float): The new reward value to incorporate into the statistics.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count     # update running mean
        delta2 = reward - self.mean
        self.M2 += delta * delta2           # update accumulator for variance
        self.step_counter += 1

        # check if it's time to reset the statistics
        if self.reset_interval and self.step_counter >= self.reset_interval:
            self.reset()

    def get_variance(self):
        """
        Returns the current variance of the rewards.

        Returns:
            float: Variance of the rewards; returns float('inf') if count < 2.
        """
        if self.count < 2:
            return float('inf')  # Variance is undefined for count < 2
        return self.M2 / self.count

    def scale(self, reward):
        """
        Scales and clips the given reward.

        Args:
            reward (float): The reward to be scaled.

        Returns:
            float: The scaled and clipped reward.
        """
        std = (self.get_variance()) ** 0.5   # Calculate standard deviation
        scaled_reward = (reward - self.mean) / (std + self.epsilon)  # Standardize reward

        # Apply clipping to the scaled reward
        scaled_reward = max(self.clip_min, min(scaled_reward, self.clip_max))
        return scaled_reward

    def reset(self):
        """
        Resets the statistics (mean, variance) to initial values.
        """
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.step_counter = 0
        print("Resetting reward statistics.")
