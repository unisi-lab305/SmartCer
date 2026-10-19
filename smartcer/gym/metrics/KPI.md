# Key performance indicators

KPIs of one run of `SmartCerEnv`, computed by `smartcer.gym.metrics.compute_kpis` from the tables
of a `ResultsTracker` (`tracker.kpis()`, `kpis.json` of a saved run). Symbols are those of
`docs/FORMULATION.md`; sums run over the T steps of the episode, U is the set of nodes and U_b the
set of nodes with a battery. Ratios with a zero denominator are 0.

They replace the 10 KPIs of 0.6.2 (`smartcer/cer_metrics/`, defect S12): its `self_consumption`
was the sharing rate below, its "TIP potential" was described as an optimisation potential, and
`net_metering` and `energy_balance` had no clear physical meaning.

## Economics

| Key | Formula | Unit |
|---|---|---|
| `welfare` | Σ_t W_t (FORMULATION section 10) | € |
| `revenue` | Σ_t Σ_u π^egr_{u,t} e^gri_{u,t} | € |
| `cost` | Σ_t Σ_u π^igr_{u,t} i^gri_{u,t} | € |
| `storage_cost` | Σ_t Σ_u π^sto_u ΔT (η^cha_u p^cha_{u,t} + p^dis_{u,t} / η^dis_u) | € |
| `incentive` | Σ_t I_t (section 9, `uniform` or `tip`) | € |
| `reward` | Σ_t R_t, for the reward type of the run (section 11) | € (minus the penalty term) |

`welfare` = `revenue` − `cost` − `storage_cost` + `incentive`.

## Energy

| Key | Formula | Unit |
|---|---|---|
| `export_kwh` | Σ_t Σ_u e^gri_{u,t} | kWh |
| `import_kwh` | Σ_t Σ_u i^gri_{u,t} | kWh |
| `shared_energy_kwh` | Σ_t E^sh_t = Σ_t min(Σ_u e^gri_{u,t}, Σ_u i^gri_{u,t}) | kWh |
| `sharing_rate` | `shared_energy_kwh` / `export_kwh`: share of the exported energy that is shared | – (fraction) |
| `coverage_rate` | `shared_energy_kwh` / `import_kwh`: share of the imported energy covered by shared energy | – (fraction) |

Shared energy is hourly and virtual: the rates are computed on the totals of the period, so they
weight each step by its energy.

## Batteries

Per battery node (`batteries.<node_id>`) and, for the totals, over U_b:

| Key | Formula | Unit |
|---|---|---|
| `charge_kwh` | Σ_t p^cha_{u,t} ΔT, energy from the node to the battery | kWh |
| `discharge_kwh` | Σ_t p^dis_{u,t} ΔT, energy from the battery to the node | kWh |
| `throughput_kwh` | Σ_t ΔT (η^cha_u p^cha_{u,t} + p^dis_{u,t} / η^dis_u), energy into and out of the cell | kWh |
| `equivalent_full_cycles` | `throughput_kwh` / (2 S^max_u); total: Σ_u throughput / (2 Σ_u S^max_u) | – |

One equivalent full cycle moves the nominal capacity into and out of the cell once.

## Constraint penalty

For the community (K_t, section 7) and per battery node (k_{u,t}):

| Key | Formula | Unit |
|---|---|---|
| `penalty_mean` | mean over the steps of K_t (k_{u,t}) | – |
| `penalty_max` | maximum over the steps of K_t (k_{u,t}) | – |
| `penalty_steps` | number of steps with K_t (k_{u,t}) > 10⁻⁹ (`PENALTY_TOLERANCE`) | steps |

The penalty measures the part of the requested action that was clipped to the dynamic bounds; the
reward weights it with σ.

## Fit Score

`smartcer.gym.metrics.fit_score(reference, candidate)`, [M24] eq. 11 (FORMULATION section 13),
compares the daily welfare series of a reference T1 (the MILP oracle) and of a controller T2
(`daily_welfare(community_table)` sums W_t per calendar day):

F = (1 − Σ_d (T2_d − T1_d)² / Σ_d (T1_d − mean(T1))²) · 100

F = 100 when the two series are equal, 0 when T2 is as far from T1 as the mean of T1, negative when
it is farther. It is undefined for a constant reference (`ValueError`).

## Other values

`steps` is the number of steps of the episode. The community table of a run also has, per step,
`obs_clipped`: the number of observation values clipped to [0, 1] (`docs/CONFIG.md` section 6).
