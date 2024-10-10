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

    def choose_actions(self, state: Dict[str, Any]) -> OrderedDict[str, Array1[np.float32]]:
        """
        Abstract method that selects control actions for each storage node.
        It must be implemented by subclasses.

        Args:
            state (Dict[str, Any]): Dictionary containing the current state of each storage node.

        Returns:
            [OrderedDict[str, NDArray[(1,), np.float32]]]:
                Ordered dictionary containing normalized control actions -> [-1,1]
        """
        actions_dict = {}
        for node_id, s in state.items():
            if node_id != 'cer':
                if node_id in self.prosumer_ids:
                    diff = s['production'] - s['consumption']
                    if diff >= 0:
                        # charge with surplus
                        charging_action = min(diff, (s['capacity'] - s['soc']))
                        charging_action = min(charging_action, s['rated_power'])
                        action = charging_action / s['capacity']
                    else:
                        # discharge to meet demand
                        discharging_action = min(-diff, max((s['soc'] - s['soc_min']), 0))
                        discharging_action = min(discharging_action, s['rated_power'])
                        action = -discharging_action / s['capacity']
                elif node_id in self.producer_ids:
                    if s['production'] > 0:
                        # charge when there is production
                        charging_action = min(s['production'], (s['capacity'] - s['soc']))
                        charging_action = min(charging_action, s['rated_power'])
                        action = charging_action / s['capacity']
                    else:
                        # discharge when there is no production
                        discharging_action = max((s['soc'] - s['soc_min']), 0)
                        discharging_action = min(discharging_action, s['rated_power'])
                        action = -discharging_action / s['capacity']
                else:
                    raise KeyError(f'Node {node_id} not belonging to any node category.')

                action = np.abs(action) if action == 0. else action  # avoid negative zeros
                actions_dict[node_id] = np.array(object=[action], dtype=np.float32)

        return collections.OrderedDict(sorted(actions_dict.items()))
