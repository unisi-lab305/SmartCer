# SmartCer

**SmartCer** is a [Gymnasium](https://gymnasium.farama.org/) environment for Italian Renewable
Energy Communities (CER, *comunità energetiche rinnovabili*). A community is a set of producers,
consumers and prosumers, some of them with a battery. In every hour it earns an incentive on the
shared energy (*energia condivisa*), the minimum between the energy its members feed into the grid
and the energy they take from it. A reinforcement learning agent controls the batteries to
maximise the social welfare of the community.

```{admonition} Beta
:class: warning
SmartCer 1.0.1 is a beta release: validation against a MILP reference trajectory is in progress.
The MILP oracle file format and the forecast error presets are provisional.
```

```bash
pip install smartcer
```

```{optional-figure} /_static/figures/overview.png
:alt: Overview of SmartCer: a community of producers, consumers and prosumers, with batteries controlled by a reinforcement learning agent through a Gymnasium environment.
:width: 90%

Overview of SmartCer.
```

## Contents

```{toctree}
:maxdepth: 1
:hidden:

installation
quickstart
concepts
configuration
FORMULATION
kpis
api
migration
citing
```

- [Installation](installation.md): requirements, `pip`, installation from source, building these docs.
- [Quickstart](quickstart.md): a first simulation on synthetic example data.
- [Concepts](concepts.md): observation, action, reward, forecasts, MILP oracle and baselines.
- [Configuration](configuration.md): scenario format, CSV files, run arguments, outputs.
- [Formulation](FORMULATION.md): the equations of the physics, economics and reward.
- [KPIs and outputs](kpis.md): the indicators and files the code produces.
- [API reference](api.md): the public modules.
- [Migrating from 0.x](migration.md): what changed since 0.6.2.
- [Citing](citing.md): how to cite SmartCer, license and acknowledgements.

## Citing

If you use SmartCer in your research, please cite it: see [Citing](citing.md).
