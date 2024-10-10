from cer_models.node_models import Producer, Prosumer
from abc import ABC, abstractmethod

from typing import Annotated, Any, Literal, TypeVar, List, Dict, OrderedDict, Union

import numpy.typing as npt
import numpy as np

DType = TypeVar('DType', bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]


class StorageController(ABC):
    """
    Abstract class that defines the interface for a Storage Controller to manage storage systems.

    Attributes:
        storage_node (Dict[str, Union[Producer, Prosumer]]): Dictionary containing the storage node objects.
        storage_node_ids (List[str]): List of storage node IDs.
        producer_ids (List[str]): List of producer node IDs.
        prosumer_ids (List[str]): List of prosumer node IDs.
    """

    def __init__(self, storage_node: Dict[str, Union[Producer, Prosumer]],
                 producer_ids: List[str], prosumer_ids: List[str]):
        """
        Initializes the StorageController abstract class.

        Args:
            storage_node (Dict[str, Union[Producer, Prosumer]]): Dictionary containing the storage node objects.
            producer_ids (List[str]): List of producer node IDs.
            prosumer_ids (List[str]): List of prosumer node IDs.
        """
        self.storage_node = storage_node
        self.storage_node_ids = list(storage_node.keys())
        self.producer_ids = producer_ids
        self.prosumer_ids = prosumer_ids

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} → storage_node_ids={self.storage_node_ids})'

    @abstractmethod
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
        pass
