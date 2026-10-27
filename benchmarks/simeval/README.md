# simeval: serving OpenWAM checkpoints to sim-policy-eval

OpenWAM speaks sim-policy-eval's model-agnostic **`sim_policy_v1`** protocol (defined by the benchmark in
sim-policy-eval `src/sim_policy_eval/policy_io/clients/sim_policy.py`; EgoSteer speaks the same one). sim-policy-eval
uses only its generic parts (`sim_observation` preprocessor, `sim_policy` client, `hand_targets` postprocessor +
its hand IK); no OpenWAM code on that side, and no sim-policy-eval code here.

```
sim-policy-eval evaluate worker                         OpenWAM (one server per GPU)
  sim_observation: SimObservation -> obs dict    ──►  server/serve.py: one Session per connection
  sim_policy client (sim_policy_v1)                     sim_policy.py: sim names -> LeRobot features
  hand_targets: {wrist, tips|fingers} -> IK      ◄──    openwam/deploy InferenceEngine
```

| file | what |
|---|---|
| `server/serve.py` | `serve.py --ckpt <run>/checkpoint_step_N.safetensors deploy.policy.steps_per_plan=K [deploy.*=...] --port P`: loads the model, then serves `sim_policy_v1` |
| `sim_policy.py` | the protocol <-> engine translation (spec, observation names, the `hand_targets` action); numpy only |
| `list_ckpts.py` | `list_ckpts.py <run_dir> --steps-per-plan K`: `<step> <ckpt> deploy.policy.steps_per_plan=K` per complete checkpoint, for `watch --list-ckpts` |

## What crosses the wire (`sim_policy_v1`)

`multiprocessing.connection`, authkey `sim-policy-eval`, pickled tuples; replies `("ok", value)` / `("error", str)`.
One connection = one episode.

- `("spec",)` -> `{"protocol": "sim_policy_v1", "chunk": 32, "execute_steps": K, "action_rate_hz": 20.0,
  "cameras": ["ego_view", "chest_view"], "action": "hand_targets", "ckpt": <served weights>}`
- `("reset", {"instruction": str, "seed": int | None})` -> `None` (seed None: the server's `deploy.policy.seed`, 42)
- `("plan", obs)` -> the next K steps' `{"wrist": (K, 2, 4, 4), "tips": (K, 2, 5, 3)}` (`fingertips_*`) or
  `{"wrist", "fingers": (K, 2, 22)}` (`unify_*`).

`wrist` is camera_from_wrist of `left_hand_wrist` / `right_hand_wrist` in the ego camera (OpenCV axes) of the
observation the plan was made from; `tips` are the five fingertips (thumb..pinky, m) in each wrist's own frame.

How `sim_policy.py` reads the observation into the engine's LeRobot features:

| `sim_policy_v1` obs | engine (training data name) |
|---|---|
| `images["ego_view"]`, `images["chest_view"]` | `observation.images.ego_view`, `observation.images.chest_view` |
| `proprio["observation.state"]` (48) | `observation.state` |
| `cameras["ego_view"]["base_from_camera"]` (4, 4) | `camera_pose.ego_view` (16, row-major) |
| `joint_pos` (56, dataset joint order) | `joint_angles` (read for `unify_*`) |
| the reset's `instruction` | `task` (the prompt, verbatim) |

## Interface profile (sim-policy-eval `configs/interfaces/openwam_v1.yaml`)

Generic parts only; the same file works for every OpenWAM yam_sharpa action mode (the server's spec says the rest):

```yaml
# Interface profile: how one model family talks to the environment. Schema: sim_policy_eval.specs.models.InterfaceSpec
# OpenWAM sparse-history yam_sharpa checkpoints served by OpenWAM's benchmarks/simeval/server/serve.py over the
# benchmark's own sim_policy_v1 protocol (policy_io/clients/sim_policy.py). The server owns the model I/O (vstack canvas,
# history buffer, proprio, sampler, unnormalize, relative decode) and returns K = 20 absolute hand-pose targets (wrist in
# ego_view + fingertips in the wrist frame, or finger joints for unify_*), which the environment's hand IK solves.
schema_version: 1
interface_id: openwam_v1
description: OpenWAM yam_sharpa policy servers (sim_policy_v1); ego + chest RGB, the 48-d hand pose and the camera poses in, 20 absolute hand-pose targets out (32 predicted), all 20 executed.
status: provisional          # confirmed after an end-to-end rollout parity run
observation_spec: rgbd_ego_chest_state48_v1   # depth is rendered but not sent
action_spec: sharpa_hand_targets_v1
preprocessor: {type: sim_observation}
client: {type: sim_policy, params: {timeout_s: 600, use_sim_seed: false}}
postprocessor: {type: hand_targets}
execution:
  policy_rate_hz: 20.0
  chunk: 32          # H (server spec "chunk" = dataloader.num_frames - 1)
  execute: 20        # K (server spec "execute_steps" = deploy.policy.steps_per_plan)
```

Watcher: `watch --interface openwam_v1 --list-ckpts "<openwam python> benchmarks/simeval/list_ckpts.py <run> --steps-per-plan 20"
--server benchmarks/simeval/server/serve.py --server-python <openwam python> --openwam-root <OpenWAM> --no-check-server`
(K must match the profile's `execute`).

## Verification

`outputs/investigations/27_deploy_refactor_parity/`: the engine reproduces the pre-refactor server element for element
(5 checkpoints); `check_server_e2e.py` runs sim-policy-eval's own `sim_observation` -> `sim_policy` -> `hand_targets`
against `serve.py` and gets the engine's plans element for element.
