HTML5 Genetic Cars
==================

A genetic algorithm car evolver in HTML5 canvas.

Inspired by BoxCar2D, uses the same physics engine (Box2D), written from scratch.

Originally published on http://rednuht.org/genetic_cars_2/

See also [Genetic Brick Cars](https://rednuht.org/genetic_brick_cars/), which
evolves cars from real LEGO® parts in 3D.

## Running

Just open `index.html` in a browser. No build step, no npm, no bundler.

For the graph-only (headless) view, open `graphs.html`.

> **Tip:** Some browsers block local file access. If needed, serve locally:
> ```
> python3 -m http.server 8000
> ```
> Then visit `http://localhost:8000`

## Flight-sim GA prototype

[`flight_sim/`](flight_sim/README.md) is a separate Python prototype that applies
this GA's ideas to tuning an altitude-hold autopilot in JSBSim. It uses
normalized genes, rank selection, and elitism. Quick start:

```
cd flight_sim && pip install -r requirements.txt
python evolve.py --config config.example.json
```

You can also run it in the browser with the
[Colab playground notebook](https://colab.research.google.com/github/PoppaMas/HTML5_Genetic_Cars/blob/flight-sim-prototype/flight_sim/playground.ipynb)
or a GitHub Codespace (`.devcontainer/` is included). See
[`flight_sim/README.md`](flight_sim/README.md) for details and example results.
