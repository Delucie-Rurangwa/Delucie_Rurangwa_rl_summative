# Delucie Rurangwa – RL Summative: Smart Charging for an E-Moto Battery-Swap Station

**Mission:** reduce waiting time at electric moto-taxi battery-swap stations in Kigali while keeping the station's electricity cost and battery wear low.

**Video:** [add your video link] · **Report:** submitted on Canvas

The agent is the station's smart charging controller. Every 10 simulated minutes it decides how the chargers run. Riders arrive at random (busier during the morning and evening rush), take a fully charged battery and leave their depleted one. If no battery is ready, they wait; after 30 minutes they leave.

![Environment](assets/environment_screenshot.png)

## Environment summary

| Item | Design |
|---|---|
| Episode | One day, 00:00–24:00, 144 steps of 10 minutes |
| Actions (Discrete 6) | `HOLD`, `START_NORMAL`, `START_FAST`, `BOOST`, `SLOW_DOWN`, `PAUSE` |
| Observation (Box 27, float32) | charge level of 10 batteries, 10 charger modes, queue length, time of day (sin/cos), tariff band, demand forecast, grid load, batteries ready |
| Reward | +1 per rider served, −0.1 per rider waiting per step, −2 per rider lost, −electricity cost (1 point = 1000 RWF), −0.05 per fast-charged kWh, −0.05 per invalid action |
| Start | 00:00, batteries at random charge levels (options: `init_soc="low"/"full"`, `demand_scale`) |
| Terminal | End of day (truncation) or 15 riders lost (station failure, −10) |
| Constraints | 8 kW grid connection; normal charging 1 kW, fast charging 2.5 kW; time-of-use tariff |

All parameters are simulation assumptions documented in `environment/custom_env.py`. Riders pay a flat price per swap; the time-of-use prices apply only to the station's own electricity bill, modelled on REG's industrial time-of-use tariff (October 2025).

## Run it

Requires [uv](https://docs.astral.sh/uv/). Everything runs on CPU.

```bash
git clone https://github.com/Delucie-Rurangwa/Delucie_Rurangwa_rl_summative.git
cd Delucie_Rurangwa_rl_summative
uv sync
uv run main.py                 # watch the best trained agent (PPO) run one day: window + terminal log
uv run main.py --algo a2c      # best agent of one algorithm: dqn, reinforce, ppo, a2c
uv run main.py --busy          # unseen busy day (+30% riders)
uv run main.py --random        # untrained random agent for comparison
uv run main.py --no-render     # terminal output only
uv run main.py --episodes 3    # several days in a row
uv run pytest                  # environment tests
```

The station window opens on the right of the screen at 70% of its width, leaving room for the terminal. To change its size, set `STATION_WIDTH`, for example `STATION_WIDTH=0.6 uv run main.py`. Close the window with the X to finish.

## Train and analyse

Each command runs the 10 hyperparameter configurations of that algorithm (150,000 environment steps each, about 1,000 simulated days), saves every model, and writes a results table to `logs/<algo>_sweep.csv`.

```bash
uv run python -m training.dqn_training               # DQN
uv run python -m training.pg_training --algo all     # REINFORCE, PPO and A2C
uv run python -m training.analysis                   # best models, plots, convergence, generalisation
```

Single runs: `--run 3`. Shorter budget: `--frames 50000`.

`training.analysis` saves the figures in `assets/` (cumulative rewards, DQN loss and policy entropy, convergence, sweep overview, generalisation) and `logs/best_models.json`, which `main.py` uses to load the best agent.

## Results

Mean reward per day of each method's best run on 20 days never used in training (greedy policy):

| Method | Best run | Reward per day | Riders served | Riders lost |
|---|---|---|---|---|
| **PPO** | 3 | **14.35 ± 7.45** | 47.4 | 3.3 |
| A2C | 9 | 12.22 ± 7.79 | 47.4 | 3.0 |
| DQN | 2 | 11.19 ± 11.68 | 45.5 | 4.2 |
| REINFORCE | 9 | 10.93 ± 9.50 | 47.1 | 3.4 |
| Rule: always start a normal charge | – | 9.58 ± 8.75 | 46.9 | 3.9 |
| Random controller | – | −35.84 | – | – |

PPO was the best and most stable method, and A2C converged fastest (about 230 days). The full hyperparameter tables are in `logs/<algo>_sweep.csv`, and the analysis is in the report.

## Project structure

```
main.py / play.py   run the best trained agent with the visualisation
environment/        custom_env.py (Gymnasium environment), rendering.py (pygame visualisation)
training/           common.py, dqn_training.py, pg_training.py (TorchRL), analysis.py
models/             saved models: dqn/ and pg/
logs/               training curves (JSON), sweep tables and summaries (CSV)
assets/             screenshot, figures, Barlow font (SIL Open Font License)
tests/              environment tests
```