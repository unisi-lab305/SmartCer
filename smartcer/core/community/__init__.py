"""
Community economics: shared energy, incentive models, welfare (docs/FORMULATION.md sections 9-10)
and the `Community` that steps its nodes together.
"""

from smartcer.core.community.community import Community, CommunityStep
from smartcer.core.community.incentives import (
    IncentiveModel,
    TipIncentive,
    UniformIncentive,
    tip_coefficient,
    tip_incentive,
    uniform_incentive,
)
from smartcer.core.community.shared_energy import shared_energy
from smartcer.core.community.welfare import WelfareBreakdown, welfare_breakdown


__all__ = [
    'Community',
    'CommunityStep',
    'IncentiveModel',
    'TipIncentive',
    'UniformIncentive',
    'WelfareBreakdown',
    'shared_energy',
    'tip_coefficient',
    'tip_incentive',
    'uniform_incentive',
    'welfare_breakdown',
]
