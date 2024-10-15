from typing import Dict, List


def compute_kpis(results_cer: Dict[str, List[float]], rounding: int = 2) -> Dict[str, float]:
    """
    Compute Key Performance Indicators (KPIs) from the results of the CER environment.

    :param (Dict[str, List[float]]) results_cer: Results of the CER environment.
    :param (int) rounding: Number of decimal places to round the KPIs.

    :return: Dictionary with the KPIs.
    """

    net_energy_expense = round(sum(results_cer['energy_expense']) - sum(results_cer['energy_revenue']), rounding)

    tip_potential = round(sum(results_cer['tip_potential']), rounding)

    tip_actual = round(sum(results_cer['tip_actual']), rounding)

    self_consumption = round((sum(results_cer['selfconsumption']) / sum(results_cer['exported_energy']) * 100),
                             rounding) if sum(results_cer['exported_energy']) > 0 else 0.0

    energy_demand = round(sum(results_cer['imported_energy']), rounding)

    gross_self_consumption_rate = round(
        (sum(results_cer['selfconsumption']) / (
                sum(results_cer['imported_energy']) + sum(results_cer['selfconsumption'])) * 100)
        if (sum(results_cer['imported_energy']) + sum(results_cer['selfconsumption'])) > 0 else 0.0,
        rounding
    )

    net_metering = round(sum(results_cer['selfconsumption']) - sum(results_cer['exported_energy']), rounding)

    cost_per_unit_energy = round(net_energy_expense / energy_demand, rounding) if energy_demand > 0 else 0.0

    revenue_per_unit_energy = round(sum(results_cer['energy_revenue']) / sum(results_cer['exported_energy']),
                                    rounding) if sum(results_cer['exported_energy']) > 0 else 0.0

    energy_balance = round(
        sum(results_cer['imported_energy']) - (
                sum(results_cer['selfconsumption']) + sum(results_cer['exported_energy'])),
        rounding
    )

    return {
        'net_energy_expense': net_energy_expense,
        'tip_potential': tip_potential,
        'tip_actual': tip_actual,
        'self_consumption': self_consumption,
        'energy_demand': energy_demand,
        'gross_self_consumption_rate': gross_self_consumption_rate,
        'net_metering': net_metering,
        'cost_per_unit_energy': cost_per_unit_energy,
        'revenue_per_unit_energy': revenue_per_unit_energy,
        'energy_balance': energy_balance
    }
