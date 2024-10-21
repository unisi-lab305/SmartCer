from smartcer.cer_models.node_models import Producer, Consumer, Prosumer

from typing import Dict, Union

import numpy as np


class Reward:
    """
    Class for reward computation. Implements different reward types.

    Attributes:
        nodes (Dict[str, Union[Producer, Consumer, Prosumer]]): Dictionary of SmartCer nodes.
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
        Compute penalty for constraint violation ∈ [0,1]. The penalty is normalized by the storage nominal power and
        multiplied by the penalty coefficient.
        For single-agent reward, returns a dictionary of penalties for each node. For collective reward, returns the
        mean of all penalties.

        Args:
            time_step (int): Current time step.

        Returns:
            Dict[str, float] or float: Penalty(s) for the current time step.
        """
        penalties, scaled_penalties = {}, {}
        for node_id, node in self.nodes.items():
            if node.storage:
                normalized_penalty = node.storage.constraint_violation[time_step] / node.storage.rated_power
                penalties[node_id] = normalized_penalty
                scaled_penalties[node_id] = normalized_penalty * self.penalty_coefficient

        if self.single_agent:
            return scaled_penalties
        else:
            return np.mean(list(penalties.values())) * self.penalty_coefficient

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
        if self.single_agent:
            rewards = {}
            for node_id, node in self.nodes.items():
                if node.storage:
                    trade = node.energy_trade[time_step]
                    rewards[node_id] = np.sign(trade) * min(0, trade)**2
            return rewards
        else:
            cumulative_reward = 0.0
            for node_id, node in self.nodes.items():
                trade = node.energy_trade[time_step]
                cumulative_reward += np.sign(trade) * min(0, trade)**2
            return cumulative_reward

    def compute_cost_reward(self, tip: float, time_step: int) -> Dict[str, float] or float:
        """
        Cost reward: reward=energy_revenue - energy_expense (+ tip for collective reward).
        For single-agent reward, returns a dictionary of rewards for each node.
        For collective reward, returns the sum of all rewards plus the total incentive (tip).

        Args:
            tip (float): Total incentive payment.
            time_step (int): Current time step.

        Returns:
            Dict[str, float] or float: Reward(s) for the current time step.
        """
        if self.single_agent:
            rewards = {}
            for node_id, node in self.nodes.items():
                if node.storage:
                    cost = node.energy_revenue[time_step] - node.energy_expense[time_step]
                    rewards[node_id] = cost
            return rewards
        else:
            cumulative_reward = 0.0
            for node_id, node in self.nodes.items():
                cost = node.energy_revenue[time_step] - node.energy_expense[time_step]
                cumulative_reward += cost
            cumulative_reward += tip
            return cumulative_reward
