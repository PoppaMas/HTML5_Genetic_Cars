# flight_sim: evolving autopilot gains with the car GA's ideas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PoppaMas/HTML5_Genetic_Cars/blob/flight-sim-prototype/flight_sim/playground.ipynb)

A first working prototype that reuses the genetic algorithm design from the
car evolver (`src/app.js`) to tune a flight controller instead of a car body.
It flies a stock **JSBSim Cessna 172 (`c172x`)** headlessly and evolves the
6 gains of an altitude-hold autopilot:

```
altitude error --(kp_alt, ki_alt, kd_alt on vertical speed)--> pitch command
pitch error    --(kp_pitch, ki_pitch, kd_pitch on pitch rate)--> elevator
```

Throttle (airspeed hold) and ailerons (wing leveler) use fixed, hand-set gains,
so the GA only tunes the longitudinal loops.

## How it maps to the car GA

| Car GA (`src/app.js`) | Here |
|---|---|
| Genes stored as floats in [0,1], decoded by `applyTypes` | Same. `genome.py` adds a `log` type for gains: `min * (max/min)**v` |
| `flatRankSelect` (geometric rank selection, p = 0.2) | `ga.flat_rank_select`, same algorithm, `--selection-p` |
| Elitism (`championLength`), parent 2 redrawn until != parent 1 | Same (`--elite`, default 2) |
| Two-point `pickParent` crossover | Uniform (default) or BLX-alpha blend (`--crossover blx`) |
| `mutateReplace`: per-gene probability, full U[0,1] redraw by default | Per-gene probability with a small Gaussian step (`--mutation-sigma`). `--mutation-mode reset` gives the original redraw |
| All cars in one Box2D world, score = distance + speed | Each individual is flown in its own JSBSim instance, in parallel processes. The cost is averaged over several seeded disturbance scenarios |
| "Health" timer ends a stuck car's run | Envelope checks end a run early: crash (<500 ft AGL), stall (<55 KCAS), pitch beyond ±30°, bank beyond ±45°, load factor outside -1 to 3.8 g, altitude error >1000 ft, or NaN |

## Task and fitness

Each evaluation flies a 90 s altitude step task. The aircraft starts trimmed at
4000 ft and 100 KCAS, the target steps to 4200 ft at t = 5 s, and steps back to
4000 ft at t = 50 s. Scenario 0 is calm air. The other scenarios (default 3 in
total) each get a seeded random steady wind of 5–25 ft/s, Gauss–Markov vertical
turbulence with an RMS of 1–4 ft/s, and one 1-cosine vertical gust of 8–15 ft/s.

Cost per scenario (lower is better):

```
cost = tracking + 2.0 * effort
tracking = mean over time of |altitude error| / 100 ft * min(t - t_step, 30 s) / 10 s   (ITAE-like)
effort   = elevator total variation (sum |delta elevator|) per second
```

The time weighting forgives the error you can't avoid right after a step and
penalizes error that lingers once the aircraft should have settled. Any
envelope violation scores `1000 + 1000 * (fraction of the run not completed)`,
so failures always rank below completed runs and failing early ranks lowest.
The GA's fitness is the mean cost over the scenarios. Selection uses only
ranks, so these scales only matter for how the terms trade off against each
other.

## Try it in the browser

- **Colab:** click the badge above, then choose *Runtime → Run all*. The
  [`playground.ipynb`](playground.ipynb) notebook clones this branch, installs
  the requirements, and runs a short evolution (pop 24, 12 generations, 2
  scenarios: about 1 minute on 2 cores). It then shows the fitness curve and
  the altitude response inline. The parameters are in the notebook's first
  code cell.
- **GitHub Codespaces:** choose *Code → Codespaces → Create codespace* on this
  branch. `.devcontainer/devcontainer.json` uses a Python 3.12 image and runs
  `pip install -r flight_sim/requirements.txt`, so you can run `cd flight_sim
  && python evolve.py ...` or open the notebook right away.

## Install locally

```bash
cd flight_sim
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # jsbsim, numpy, matplotlib
```

The `jsbsim` wheel ships the stock aircraft, so nothing else is needed.

## Run

```bash
python evolve.py --config config.example.json        # the run in results/example (~2.5 min on 8 cores)
python evolve.py --pop-size 24 --generations 10 --seed 7 --out results/quick
python evolve.py --help                               # all options
python test_flight_ga.py                              # quick self-checks (or: pytest)
python plot_results.py results/example                # re-draw plots for a finished run
```

Settings are applied in this order, with later ones winning: built-in defaults,
then `--config file.json`, then explicit flags. A run is deterministic for a
given config and seed. The worker count doesn't change results, because
simulations are deterministic, results are collected in population order, and
all GA randomness comes from one seeded `numpy` generator. Elites aren't
re-flown: since the sim is deterministic, their cost is cached.

## Outputs (`--out` directory)

- `fitness_history.csv`: one row per generation with the best, mean, and
  median cost, the number of individuals with at least one envelope failure,
  the number of new evaluations, the elapsed time, and the best gains.
- `best_gains.json`: the best gains (with units), the normalized genome, the
  per-scenario cost breakdown and status, the gen-0 best gains for comparison,
  the scenarios used, and the full config.
- `best_response.png`: the best controller's altitude vs the target in every
  scenario, with pitch and elevator below. The gen-0 best on the calm scenario
  is drawn dotted for contrast.
- `fitness_curve.png`: best and median cost per generation (log scale).

## Example result (`results/example`, seed 1, pop 48, 40 generations, 3 scenarios)

The CSV and JSON are committed. The two PNG plots are not, because they had to
be committed with a text-only tool. To regenerate them (about 5 s,
deterministic), run `python plot_results.py results/example`, or re-run the
config above, which reproduces the CSV and JSON exactly.

| | mean cost |
|---|---|
| hand-picked gains (kp_alt 0.05, ki_alt 0.001, kd_alt 0.3, kp_pitch 0.05, ki_pitch 0.01, kd_pitch 0.02) | 0.654 |
| best of generation 0 (48 random genomes) | 0.319 |
| best after 40 generations | **0.186** (calm 0.169, wind/gust cases 0.205 and 0.182) |

The median cost dropped from 4.31 to 0.19. In generation 0, 13 of the 48
genomes broke the envelope in at least one scenario. By generation 4 failures
were rare (one child in generation 5), and from generation 6 on there were
none. Evolved gains:

| gene | value | units |
|---|---|---|
| kp_alt | 0.500 (upper bound) | deg pitch per ft |
| ki_alt | 1.92e-5 | deg per ft·s |
| kd_alt | 0.693 | deg per ft/s |
| kp_pitch | 0.0238 | elevator per deg |
| ki_pitch | 0.00186 | elevator per deg·s |
| kd_pitch | 0.00895 | elevator per deg/s |

How to read this: the evolved controller climbs and descends at the pitch
command limit (+12° / −8° relative to trim) and captures the new altitude
within about 5 ft, with no large overshoot. The gen-0 best is slower and
overshoots by about 20 ft (dotted line in the plot). `kp_alt` sits on its upper
bound, so the outer loop is effectively "max pitch until about 25 ft from
target". Widening that range or lowering the pitch limit would change the
character of the solution.

## Caveats / next steps

- This is a prototype. The cost weights, envelope limits, and pitch-command
  limits are hand-picked constants at the top of `sim.py`. The evolved
  behavior depends on them, and climbing at about 15° pitch is aggressive for a
  C172.
- Only three scenarios are used, so the controller can still overfit them. For
  robustness, raise `--scenarios` or validate on a different
  `--scenario-seed`.
- The control loop runs at the 120 Hz FDM rate, with no sensor noise, delay,
  or actuator model.
- Gene ranges are guesses. Two genes converged to a bound (`kp_alt` high,
  `ki_alt` low). Best cost was still creeping down at generation 40.
- Possible next steps from the write-up: evolve heading/roll gains too,
  co-evolve airframe parameters, or try the unused simulated-annealing
  refinement (`manageRoundSA`) on the best genome.
