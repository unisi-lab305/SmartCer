from storage_controllers.rbc_controller import StandardRBC
from cer_metrics.scores import compute_kpis
from cer_env import Cer

from tqdm import tqdm
import collections
import numpy as np


def compute_baselines(cer: Cer, seed: int, no_storage: bool = True, rbc: bool = None) -> None:
    """
    Compute baseline results for the CER environment.
    Results are saved in the CER data folder.

    :param (Cer) cer: CER environment.
    :param (int) seed: Seed for the random number generator.
    :param (bool) no_storage: If True, the baseline with no storage is computed.
    :param (bool) rbc: If True, the baseline with the Rule-Based Controller is computed.

    :return: None
    """

    if no_storage:
        actions = {node_id: np.array(object=[0.], dtype=np.float32) for node_id in cer.storage_nodes_ids}
        actions = collections.OrderedDict(sorted(actions.items()))

        results = {}
        _, _ = cer.reset(seed=seed)
        for _ in tqdm(range(cer.sim_hours), desc="No Storage Simulation Progress"):
            _, _, terminated, truncated, _, _ = cer.step(actions)
            done = terminated or truncated
            if done:
                results = cer.get_sim_results()
                cer.reset()
        cer.save_results(results=results, output_dir='baseline_no_storage', save_mode='csv')

    if rbc:
        rbc_controller = StandardRBC(
            storage_node=cer.storage_nodes, producer_ids=cer.producers_ids, prosumer_ids=cer.prosumers_ids)

        results = {}
        _, info = cer.reset(seed=seed)
        for _ in tqdm(range(cer.sim_hours), desc="RBC Simulation Progress"):
            actions = rbc_controller.choose_actions(info)
            _, _, terminated, truncated, new_info, _ = cer.step(actions)
            done = terminated or truncated
            if done:
                results = cer.get_sim_results()
                cer.reset()
            else:
                info = new_info
        cer.save_results(results=results, output_dir='baseline_rbc', save_mode='csv')