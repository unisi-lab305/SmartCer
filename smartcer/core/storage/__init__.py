"""
Storage systems controlled by a normalised action (docs/FORMULATION.md sections 4-7).
"""

from smartcer.core.storage.base import StorageSystem
from smartcer.core.storage.battery import Battery, BatteryStep


__all__ = ['Battery', 'BatteryStep', 'StorageSystem']
