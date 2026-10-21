# Reproducing the Go2–SO-101 fault-tolerant locomotion experiment

## What is included

- Task ID: `LameDog-Velocity-Flat-Go2So101` (`ManagerBasedRLEnv`, RSL-RL PPO).
- Robot: Go2 plus SO-101 arm; front-left hip, thigh, and calf actuators have zero stiffness, damping, and effort.
  Their links and colliders remain in the USD. The nine joints on the other three legs are policy actions; the six
  arm joints have drives but are not policy actions.
- Physics: Newton MJWarp; flat plane, contact sensor, 0.005 s physics timestep, four physics steps per policy action.
- Included assets: `src/lame_dog/assets/data/robots/go2_so101.usda` and its relative
  `payloads/Attachment/so101.usd` layer. These are composition layers, not copies of the upstream robot meshes.
  Loading them requires access to the two NVIDIA-hosted USD references embedded in the layers.
- Included checkpoints: `checkpoints/model_4999.pt` (end of initial low-speed phase),
  `checkpoints/model_25700.pt` (selected point in the low-speed continuation), and
  `checkpoints/model_35699.pt` (end of the full-speed continuation). None requires Git LFS.
- `results/continuation_scalars.csv` contains five TensorBoard scalar series for every iteration from 25700 to
  35699, without machine-specific log paths. It is training telemetry, not held-out evaluation data.

This work was run on Linux, Python 3.12.15, uv 0.13.1, NVIDIA RTX 4090, driver 560.35.03, using Isaac Lab commit
`0c114a24a3b9e854d299d3527ada507139fc4e40`. The Isaac Lab source checkout had local changes to dependency
manifests and comments in `unitree.py`; no task or physics implementation changes were observed in that checkout.
The local project `uv.lock` records the dependency versions. GPU/driver or package changes may alter trajectories.

For integrity checking, the SHA-256 values are:

| Checkpoint | SHA-256 |
| --- | --- |
| `model_4999.pt` | `f843946af2dc23a3643af9e7cc741132a6ed15505dd17315f750a389c783c0d8` |
| `model_25700.pt` | `9550e2f0cc763f90de10b4251e40e09aeb94de08e6c6a35b73c0e2fe096889c2` |
| `model_35699.pt` | `664b9c73a556259e739c2f52f142bfdcb19e6bf30db2969d6dd9a35e0523fb4f` |

## Install

Place both repositories beside each other; `pyproject.toml` uses editable paths into `../IsaacLab`:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 0c114a24a3b9e854d299d3527ada507139fc4e40
git clone https://github.com/LightLNT/lame_dog.git
cd lame_dog
uv sync --locked
```

The project uses the default Newton stack; a local Isaac Sim installation is not required for these commands.
The packaged USD layers still resolve two external NVIDIA asset URLs, so the first scene load needs network access.
If you put the Isaac Lab checkout elsewhere, update the `[tool.uv.sources]` paths in `pyproject.toml` and regenerate
the lockfile before syncing.

## Verify the environment

```bash
uv run --no-sync python scripts/list_envs.py
uv run --no-sync python -m pytest tests/test_registration.py
uv run --no-sync isaaclab random_agent --task LameDog-Velocity-Flat-Go2So101 --num_envs 4
```

The first two commands check installation and registration. The random-agent command starts a physical scene; stop
it with Ctrl-C after it steps. These checks do not establish that a trained policy walks successfully.

## Train

The agent configuration uses PPO with a three-layer 128-unit actor and critic, 24 steps per environment per iteration,
and seed 42. The runs used 128 parallel environments and nine position actions with a scale of 0.25 rad. The failed
leg stayed passive throughout. The original checkpoint lineage has three stages:

1. `LameDog-Velocity-Flat-Go2So101-LowSpeed`: 5000 iterations from scratch. Commands were
   `vx ∈ [-0.3, 0.3] m/s`, `vy ∈ [-0.2, 0.2] m/s`, and yaw rate `[-0.3, 0.3] rad/s`. The push event and foot-air-time
   reward were disabled; success thresholds were 0.1 m/s and 0.1 rad/s. Output: `model_4999.pt`.
2. The same low-speed task continued from `model_4999.pt`. The original run requested 25,000 more iterations; its
   selected intermediate checkpoint was `model_25700.pt`.
3. `LameDog-Velocity-Flat-Go2So101` continued from `model_25700.pt` for 10,000 iterations. Commands expanded to
   `vx, vy ∈ [-1, 1] m/s` and yaw rate `[-1, 1] rad/s`, random velocity pushes were enabled, and a powered-foot
   air-time reward was added. Output: `model_35699.pt`.

To follow that sequence, use these commands. The first two use the low-speed task ID:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task LameDog-Velocity-Flat-Go2So101-LowSpeed --num_envs 128 \
  --max_iterations 5000 --run_name low_speed_5k

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task LameDog-Velocity-Flat-Go2So101-LowSpeed --num_envs 128 \
  --max_iterations 25000 --checkpoint checkpoints/model_4999.pt \
  --run_name low_speed_continue
```

The second command may be stopped after it writes a checkpoint at iteration 25700; the supplied checkpoint records
the exact state selected in the original run. Then switch to the full-speed task:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task LameDog-Velocity-Flat-Go2So101 --num_envs 128 \
  --max_iterations 10000 --checkpoint checkpoints/model_25700.pt \
  --run_name full_speed_push_from_25700
```

`--max_iterations` for a resumed run counts *additional* learning iterations. As an alternative experiment, you
can train the current full-speed task from scratch with, for example, 30,000 iterations, but that does **not**
reproduce the provided final checkpoint's low-speed-to-full-speed curriculum.

Training saves logs and checkpoints under `logs/rsl_rl/lame_dog_go2_so101_flat/`. Exact floating-point equality with
the included checkpoint is not expected across hardware, software, or nondeterministic simulation runs.

## Replay

For remote machines, Viser provides a browser-based visualizer. Install its optional dependency and play the
included final policy:

```bash
uv sync --locked --extra viser
uv run --no-sync isaaclab play --rl_library rsl_rl \
  --task LameDog-Velocity-Flat-Go2So101 --num_envs 1 \
  --checkpoint checkpoints/model_35699.pt --viz viser
```

The playback path uses the task's `play_mode()`, which disables observation corruption and random pushes; it is not
the same distribution as training. If Viser cannot be accessed from a remote browser, use SSH port forwarding or
play locally with a supported visualizer.

## Interpreting the reported training result

In the original continuation run, the final TensorBoard training values at iteration 35699 were approximately
`mean_reward = 15.80`, `mean_episode_length = 821` policy steps, horizontal velocity error `0.103 m/s`, yaw-rate
error `0.197 rad/s`, and logged `success_rate = 1.0`. The success metric is calculated on reset episodes whose
*episode-mean* horizontal and yaw errors are below 0.5 m/s and 0.4 rad/s respectively. These are **training-log
metrics**, not a separate held-out evaluation or physical-robot result. Replays and independent multi-seed evaluation
are needed before claiming robust performance.

The asset loads external Go2 and SO-101 references. This repository distributes only its small composition layers;
it does not redistribute those source robots' meshes or textures. Follow the upstream asset terms when using them.
