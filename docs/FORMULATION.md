# SmartCer formulation

**Version:** 1.0 (target of the 1.0 refactoring)
**Status:** authoritative reference for the physics and the economics implemented in `smartcer/core/` and for the reward in `smartcer/gym/rewards/`. Any change to those equations must update this file in the same commit.

## 1. Sources

- [S22] M. Stentati, S. Paoletti, A. Vicino, *Optimization of energy communities in the Italian incentive system*, IEEE PES ISGT-Europe 2022. Battery dynamics, node energy balance, social welfare. This is the model of the MILP oracle used in training.
- [M24] L. Guiducci, G. Palma, M. Stentati, A. Rizzo, S. Paoletti, *A Reinforcement Learning approach to the management of Renewable Energy Communities*, MECO 2024. Action definition, dynamic action bounds, constraint penalty, reward, actor state, rule-based controller, Fit Score.

Where the two papers differ, the battery dynamics of [S22] prevail, because the optimal actions computed by the MILP must be executable in the environment. Deviations are listed in section 14.

## 2. Community, nodes and time

- A community has a set U of nodes. Node types:
  - **producer**: generation g, no load, optional battery;
  - **consumer**: load l, no generation, optional battery (with the charge rule of section 6 a consumer battery can never charge);
  - **prosumer**: generation g and load l, optional battery.
- Time is discrete with step Δt in hours (default 1.0). The CSV frequency must match Δt.
- Internal units: power in kW, energy in kWh, prices in €/kWh.
- CSV units are declared per scenario with `units`: `kWh` (energy per step, default, the 0.6.2 format and the papers) or `kW` (average power over the step). The loader converts everything to per-step energies g_{u,t}, l_{u,t} in kWh.

## 3. Symbols

| Symbol | Meaning | Unit | Config key or CSV column |
|---|---|---|---|
| g_{u,t} | generation of node u in step t | kWh | `production` |
| l_{u,t} | load of node u in step t | kWh | `consumption` |
| π^egr_{u,t} | sell price | €/kWh | `sell_price` (alias `price_exp`) |
| π^igr_{u,t} | buy price | €/kWh | `buy_price` (alias `price_imp`) |
| π^inc_t | unit incentive on shared energy (model `uniform`) | €/kWh | `incentive.unit_value` (constant or CSV column) |
| π^sto_u | unit cost of storage usage | €/kWh | `storage.usage_cost` (default 0) |
| S^max_u | battery capacity | kWh | `storage.capacity` |
| S^min_u | minimum state of charge | kWh | `storage.soc_min` as a fraction of capacity |
| S^ini_u | initial state of charge | kWh | `storage.soc_initial` as a fraction of capacity (default = `soc_min`) |
| P^cha_u, P^dis_u | maximum charge and discharge power, node side | kW | `storage.power_charge_max`, `storage.power_discharge_max` (0.6.2: `rated_power` for both) |
| η^cha_u, η^dis_u | charge and discharge efficiency | – | `storage.efficiency_charge`, `storage.efficiency_discharge` (0.6.2: `efficiency` for both) |
| α_u | self-discharge per hour (fraction of the state of charge lost in one hour) | – | `storage.alpha` (default 0; 0.6.2: `alfa`, per hourly step) |
| σ | penalty coefficient | – | `reward.penalty_coefficient` (default 10, as in [M24]) |
| s_{u,t} | state of charge at the end of step t | kWh | state |
| a_{u,t} | normalised action | – | action |

## 4. Action and battery power

One action per node with a battery, a_{u,t} ∈ [−1, 1], in the fixed order `storage_node_ids`. Positive values charge, negative values discharge ([M24] Sec. III-A, with separate limits as in [S22] eq. 4-5):

$$
p^{cha}_{u,t} = \max(0, a_{u,t})\, P^{cha}_u, \qquad p^{dis}_{u,t} = \max(0, -a_{u,t})\, P^{dis}_u
$$

Charge and discharge are never simultaneous by construction.

## 5. Battery dynamics

[S22] eq. 6-8, with optional self-discharge applied before the step and clamped at S^min (α = 0 reproduces the papers). α_u is defined per hour and converted to the step, so that the loss per hour does not depend on ΔT:

$$
\alpha^{step}_u = 1 - (1 - \alpha_u)^{\Delta T}, \qquad \tilde s_{u,t-1} = \max\big(S^{min}_u,\ (1 - \alpha^{step}_u)\, s_{u,t-1}\big) = \max\big(S^{min}_u,\ (1 - \alpha_u)^{\Delta T}\, s_{u,t-1}\big)
$$

$$
s_{u,t} = \tilde s_{u,t-1} + \eta^{cha}_u\, p^{cha}_{u,t}\, \Delta T - \frac{1}{\eta^{dis}_u}\, p^{dis}_{u,t}\, \Delta T, \qquad S^{min}_u \le s_{u,t} \le S^{max}_u, \qquad s_{u,0} = S^{ini}_u
$$

Both power limits act on the node side: in charge the node supplies p^cha ΔT and the cell stores η^cha p^cha ΔT; in discharge the cell releases p^dis ΔT / η^dis and the node receives p^dis ΔT. No degradation is modelled.

## 6. Dynamic action bounds

Derived from section 5 and from the charge rule (no charging from the grid):

$$
A^{upper}_{u,t} = \frac{1}{P^{cha}_u} \min\Big( P^{cha}_u,\ \frac{c_{u,t}}{\Delta T},\ \frac{S^{max}_u - \tilde s_{u,t-1}}{\eta^{cha}_u\, \Delta T} \Big)
$$

$$
A^{lower}_{u,t} = \frac{1}{P^{dis}_u} \min\Big( P^{dis}_u,\ \frac{\eta^{dis}_u\, (\tilde s_{u,t-1} - S^{min}_u)}{\Delta T} \Big)
$$

The charge cap c_{u,t} depends on `storage.charge_limit`:

- `production` (default, [M24] eq. 1): c_{u,t} = g_{u,t};
- `surplus` (option, pending the DIISM answer): c_{u,t} = max(0, g_{u,t} − l_{u,t}).

Feasible actions lie in [−A^lower_{u,t}, A^upper_{u,t}].

## 7. Clipping and constraint penalty

An action outside the feasible interval is replaced by the nearest bound before it is applied, and penalised ([M24] eq. 6-7):

$$
k_{u,t} = \begin{cases} \max(0,\ a_{u,t} - A^{upper}_{u,t}) & a_{u,t} > 0 \\ \max(0,\ |a_{u,t}| - A^{lower}_{u,t}) & a_{u,t} < 0 \\ 0 & a_{u,t} = 0 \end{cases} \qquad K_t = \frac{1}{|U_b|} \sum_{u \in U_b} k_{u,t}
$$

where U_b is the set of nodes with a battery. Equivalently, k_{u,t} is the clipped power in kW divided by P^cha_u (charge) or P^dis_u (discharge).

## 8. Node energy balance

[S22] eq. 13 and 16-17. The net energy exchanged with the grid, and its split into export and import:

$$
n_{u,t} = g_{u,t} - l_{u,t} + p^{dis}_{u,t}\, \Delta T - p^{cha}_{u,t}\, \Delta T, \qquad e^{gri}_{u,t} = \max(0, n_{u,t}), \qquad i^{gri}_{u,t} = \max(0, -n_{u,t})
$$

The complementarity constraint e^gri · i^gri = 0 of [S22] eq. 12 holds by construction. Contracted export and import limits ([S22] eq. 10-11) are not modelled.

## 9. Shared energy and incentive

Shared energy in step t ([S22] eq. 15):

$$
E^{sh}_t = \min\Big( \sum_{u} e^{gri}_{u,t},\ \sum_{u} i^{gri}_{u,t} \Big)
$$

Incentive I_t, selected with `incentive.model`:

- `uniform` (default for 1.0 scenarios, [S22] and [M24]):

$$
I_t = \pi^{inc}_t\, E^{sh}_t
$$

- `tip` (the 0.6.2 per-plant model, default for scenarios in the 0.6.2 format). Every node u with a `photovoltaic` section has the coefficient

$$
c_{u,t} = \Big( \min\big(TIP^{th}_u,\ TIP^{base}_u + \max(0,\ 0.180 - P^z_{u,t})\big) + f^{zone}_u \Big) \Big( 1 - 0.5\, \frac{PNRR_u}{40} \Big)
$$

  and every node without one (consumers, and any node without a plant) has c_{u,t} = 0. The incentive is

$$
I_t = \frac{E^{sh}_t}{\sum_u e^{gri}_{u,t}} \sum_u c_{u,t}\, e^{gri}_{u,t} \qquad (I_t = 0 \text{ if } \textstyle\sum_u e^{gri}_{u,t} = 0)
$$

  where both sums run over all nodes, as in 0.6.2: the shared energy is valued at the export-weighted average coefficient of all exported energy. Energy exported by the battery of a node without a plant enters the denominator with c = 0, so it lowers the average coefficient (and the incentive per kWh shared).

  P^z_{u,t} is approximated by the node's sell price π^egr_{u,t}; PNRR_u ≤ 40 (percentage of capital grant); TIP^base_u, TIP^th_u, f^zone_u in €/kWh from `photovoltaic` (`tip_base`, `tip_threshold`, `fi_zone`, `pnrr`). With equal coefficients c_{u,t} = π^inc_t for every exporting node the two models coincide.

## 10. Social welfare

[M24] eq. 3, that is [S22] eq. 14 without the flexibility terms:

$$
W_t = \sum_{u} \Big[ \pi^{egr}_{u,t}\, e^{gri}_{u,t} - \pi^{igr}_{u,t}\, i^{gri}_{u,t} - \pi^{sto}_u\, \Delta T \Big( \eta^{cha}_u\, p^{cha}_{u,t} + \frac{1}{\eta^{dis}_u}\, p^{dis}_{u,t} \Big) \Big] + I_t
$$

The storage term charges the energy that enters or leaves the cell. Producers have no buy price requirement: under the charge rule they never import.

## 11. Reward

Default reward type `welfare` ([M24] eq. 8):

$$
R_t = W_t - \sigma\, K_t
$$

Other reward types, kept for compatibility and reported per node in `info`:

- `profit`: per-node π^egr e^gri − π^igr i^gri − storage cost, plus the incentive for the collective variant;
- `energy` (0.6.2 legacy): −(i^gri_{u,t})² per node.

With m_{u,t} the per-node term above, each of these types has two variants:

- collective: R_t = M_t − σ K_t, with M_t = Σ_u m_{u,t} over all nodes, plus I_t for `profit` (so that the collective `profit` equals `welfare`);
- per node (`reward_per_node`, the 0.6.2 `single_agent`): r_{u,t} = m_{u,t} − σ k_{u,t} for every u ∈ U_b, reported in `info`, and the scalar reward is R_t = Σ_{u ∈ U_b} r_{u,t}.

`welfare` has no per-node variant.

With N = |U_b| batteries the penalty of the per-node variant is Σ_{u ∈ U_b} σ k_{u,t} = N σ K_t, N times the collective one, and the main terms differ too: collective and per-node rewards, including their training curves, are not numerically comparable.

No running reward scaling inside the environment; scaling, if wanted, is applied by a wrapper or by the training library.

## 12. Information timing

Controlled by `observe_current_step`:

- `True` (as in [M24]): when the agent chooses a_{u,t}, the observation contains g_{u,t}, l_{u,t}, the prices of step t and the bounds A^upper_{u,t}, A^lower_{u,t} of section 6.
- `False` (causal): the observation contains g and l of step t − 1; prices of step t remain observable (day-ahead sell prices, fixed buy prices); the observed A^upper is computed without the charge-cap term so that it does not reveal g_{u,t}. The environment still applies the full bounds of section 6 when clipping.

Forecasts, when added, cover steps t + 1 … t + H with `True` and t … t + H − 1 with `False`.

## 13. Baseline controller and comparison metric

**Rule-based controller** ([M24] Algorithm 1), computed in energy (node side) and converted to an action. It reads g and l of step t. The energies a battery can take and deliver in step t are those of the dynamic bounds of section 6 (power limits, charge cap, efficiencies, S^min, S^max, self-discharge):

$$
E^{cha}_{u,t} = A^{upper}_{u,t}\, P^{cha}_u\, \Delta T, \qquad E^{dis}_{u,t} = A^{lower}_{u,t}\, P^{dis}_u\, \Delta T
$$

A charge energy x gives the action x / (P^cha_u ΔT), a discharge energy x the action −x / (P^dis_u ΔT). Every action lies in [−A^lower_{u,t}, A^upper_{u,t}], so the controller is never clipped or penalised (K_t = 0). Two modes:

- **`node`** (each battery follows its own node; intended for single-node use), with d = g_{u,t} − l_{u,t}:
  - if u is a producer and g_{u,t} = 0: discharge E^dis_{u,t} (rule of 0.6.2; with d = g a producer would never discharge);
  - otherwise, if d ≥ 0: charge min(d, E^cha_{u,t});
  - otherwise: discharge min(−d, E^dis_{u,t}).
- **`community`** (the batteries follow the community balance; class A reference baseline), with B_t = Σ_{u ∈ U} g_{u,t} − Σ_{u ∈ U} l_{u,t} over all nodes:
  - if B_t > 0: every battery charges f E^cha_{u,t} with f = min(1, B_t / Σ_{v ∈ U_b} E^cha_{v,t}), that is min(B_t, Σ E^cha) split pro rata to E^cha;
  - if B_t < 0: every battery discharges f E^dis_{u,t} with f = min(1, −B_t / Σ_{v ∈ U_b} E^dis_{v,t});
  - if B_t = 0, or the sum is 0: idle.

The rule-based controller is a reference, not a lower bound: it is not required to beat idle batteries.

**Fit Score** ([M24] eq. 11) between the daily welfare series of a reference T1 (the MILP oracle) and of a controller T2:

$$
F = \Big( 1 - \frac{\sum_d (T_{2,d} - T_{1,d})^2}{\sum_d (T_{1,d} - \overline{T_1})^2} \Big) \cdot 100
$$

## 14. Declared deviations from the papers

| Item | Papers | SmartCer 1.0 | Reason |
|---|---|---|---|
| Dynamic bounds | [M24] eq. 1-2 use (c − SoC)/ΔT and SoC/ΔT, without η or S^min | derived from [S22] dynamics, with η and S^min (section 6) | the [M24] bounds allow a discharge that drives the state of charge below S^min (defect of 0.6.2) |
| Power rating | [M24]: a single rated power r_u | separate P^cha_u and P^dis_u as in [S22]; P^cha = P^dis = r_u recovers [M24] | generality at no cost |
| Incentive | uniform π^inc_t | `uniform` (default) and `tip` | the default must match the MILP objective |
| SoC in the actor state | [M24] Tab. II lists a "periodic" encoder | scaled to [S^min, S^max] | likely typo in the table |
| End-of-day constraints | [M24]: batteries empty at the start and end of each day in the MILP | none; S^end not modelled | class A decision of 14/10/2026 |
| Contracted grid limits | [S22] eq. 10-11 | not modelled | not binding with the charge rule and fixed loads |
| Demand and generation flexibility | [S22] eq. 1-3, costs π^reg, π^fle | not modelled | shiftable loads are handled by the DIISM, not by RL |

## 15. Pending confirmations from the DIISM

Defaults used until the answers arrive (questions in the project knowledge base, `SmartCer_domande_DIISM.md`):

| Item | Default |
|---|---|
| Charge rule for prosumers | `production` |
| Power limits | node side for both charge and discharge |
| π^inc for the Molise scenario | constant, value from the scenario config |
| π^sto | 0.01 €/kWh in the Molise scenario config; 0 when absent |
| S^ini | S^min |
| Self-discharge | none |
| Contracted grid limits | not active |
| Welfare terms beyond eq. 14 | none |
