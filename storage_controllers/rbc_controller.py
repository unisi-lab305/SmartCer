from storage_controllers.abc_controller import StorageController

from typing import Annotated, Any, Literal, TypeVar, Dict, OrderedDict

import numpy.typing as npt
import numpy as np

import collections

DType = TypeVar('DType', bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]


class StandardRBC(StorageController):
    """
    Class that implements a Rule-Based Controller (RBC) for managing storage systems.
    Extends the abstract class StorageController.
    """

    def choose_actions(self, observation: Dict[str, Any]) -> OrderedDict[str, Array1[np.float32]]:
        """
        Abstract method that selects control actions for each storage node.
        It must be implemented by subclasses.

        Args:
            observation (Dict[str, Any]): Dictionary containing the current state of each storage node.

        Returns:
            [OrderedDict[str, NDArray[(1,), np.float32]]]:
                Ordered dictionary containing normalized control actions -> [-1,1]
        """
        actions_dict = {}
        for node_id, obs in observation.items():
            if node_id != 'cer':

                # features not in observation vector
                capacity = self.storage_node[node_id].storage.capacity
                rated_power = self.storage_node[node_id].storage.rated_power
                soc_min = self.storage_node[node_id].storage.soc_min

                if node_id in self.prosumer_ids:
                    diff = obs['production'] - obs['consumption']
                    if diff >= 0:
                        # charge with surplus
                        charging_action = min(diff, (capacity - obs['soc']))
                        charging_action = min(charging_action, rated_power)
                        action = charging_action / capacity
                    else:
                        # discharge to meet demand
                        discharging_action = min(-diff, max((obs['soc'] - soc_min), 0))
                        discharging_action = min(discharging_action, rated_power)
                        action = -discharging_action / capacity
                elif node_id in self.producer_ids:
                    if obs['production'] > 0:
                        # charge when there is production
                        charging_action = min(obs['production'], (capacity - obs['soc']))
                        charging_action = min(charging_action, rated_power)
                        action = charging_action / capacity
                    else:
                        # discharge when there is no production
                        discharging_action = max((obs['soc'] - soc_min), 0)
                        discharging_action = min(discharging_action, rated_power)
                        action = -discharging_action / capacity
                else:
                    raise KeyError(f'Node {node_id} not belonging to any node category.')

                action = np.abs(action) if action == 0. else action  # avoid negative zeros
                actions_dict[node_id] = np.array(object=[action], dtype=np.float32)

        return collections.OrderedDict(sorted(actions_dict.items()))
