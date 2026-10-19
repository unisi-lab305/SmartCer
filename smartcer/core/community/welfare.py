"""
Social welfare of a community step (docs/FORMULATION.md section 10, [M24] eq. 3).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WelfareBreakdown:
    """
    Welfare of one step and its components.

    Attributes:
        revenue: Σ_u π^egr_u e^gri_u (€).
        cost: Σ_u π^igr_u i^gri_u (€).
        storage_cost: Σ_u π^sto_u · cell throughput (€).
        incentive: Incentive on shared energy I (€).
        shared_energy_kwh: Shared energy E^sh (kWh).
        export_kwh: Σ_u e^gri_u (kWh).
        import_kwh: Σ_u i^gri_u (kWh).
        total: Welfare W = revenue - cost - storage_cost + incentive (€).
    """

    revenue: float
    cost: float
    storage_cost: float
    incentive: float
    shared_energy_kwh: float
    export_kwh: float
    import_kwh: float
    total: float


def welfare_breakdown(
    revenue: float,
    cost: float,
    storage_cost: float,
    incentive: float,
    shared_energy_kwh: float,
    export_kwh: float,
    import_kwh: float,
) -> WelfareBreakdown:
    """
    Assemble the welfare breakdown, W = revenue - cost - storage cost + incentive.

    Args:
        revenue: Revenue from exported energy (€).
        cost: Cost of imported energy (€).
        storage_cost: Storage usage cost (€).
        incentive: Incentive on shared energy (€).
        shared_energy_kwh: Shared energy (kWh).
        export_kwh: Total exported energy (kWh).
        import_kwh: Total imported energy (kWh).

    Returns:
        WelfareBreakdown: The components and the total.
    """
    return WelfareBreakdown(
        revenue=revenue, cost=cost, storage_cost=storage_cost, incentive=incentive,
        shared_energy_kwh=shared_energy_kwh, export_kwh=export_kwh, import_kwh=import_kwh,
        total=revenue - cost - storage_cost + incentive,
    )
