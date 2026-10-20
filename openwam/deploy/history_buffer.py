"""Per-episode buffer of past observations for sparse-history checkpoints.

A sparse-history model conditions on H frames at fixed offsets before frame t (``dataloader.history_offsets``,
in action steps of the model's action chunk, e.g. [-80, -60, -40, -20, 0] at 20 Hz = t-4s .. t). Deploy plans a
chunk, the robot / simulator executes its first K steps, then sends the next observation: observations arrive
every K steps (``action_fps / K`` Hz). Every history offset must then be a multiple of K, so each history frame
is exactly one of the buffered observations; anything else is a config error, raised at construction.

Observation j of an episode is step ``j * K``; its history steps follow the training dataset's rule
(``history_rows``: steps before the episode start repeat the first observation).
"""

from collections import deque

from openwam.dataloader.utils.sparse_history import history_rows


class ObsHistoryBuffer:
    """Ring buffer of one episode's preprocessed observations, one per plan.

    Args:
        history_offsets: H ints, increasing, <= 0, last one 0 (the checkpoint's ``dataloader.history_offsets``),
            in action steps.
        action_fps: int, rate of the model's action chunk = the robot's control steps (20).
        steps_per_obs: int K, action steps executed per plan, i.e. between two observations.
    """

    def __init__(self, history_offsets, *, action_fps: int, steps_per_obs: int):
        self.history_offsets = [int(o) for o in history_offsets]
        self.action_fps, self.steps_per_obs = int(action_fps), int(steps_per_obs)
        offs, k = self.history_offsets, self.steps_per_obs
        if not offs or offs[-1] != 0 or sorted(offs) != offs:
            raise ValueError(f"history_offsets must increase and end at 0, got {offs}")
        if k <= 0 or self.action_fps % k:
            raise ValueError(
                f"steps per plan K={k} must divide action_fps={self.action_fps} (observations every {k} steps = "
                f"{self.action_fps}/{k} Hz must be a whole number per second)"
            )
        if any(o % k for o in offs):
            raise ValueError(f"every history offset {offs} must be a multiple of K={k} steps, else a history frame "
                             "falls between two observations")
        self._obs = deque(maxlen=1 - offs[0] // k)
        self._num_obs = 0  # observations appended since the last reset

    def append(self, obs: dict) -> None:
        """Record the observation of the current plan.

        Args:
            obs: preprocessed observation dict (``image`` canvas, ``state``, ``camera_pose`` ...).
        """
        self._obs.append(obs)
        self._num_obs += 1

    def window(self) -> list:
        """The H history observations of the current plan, oldest first.

        Returns: list of H observation dicts; entry H-1 is the latest appended observation.
        """
        if self._num_obs == 0:
            raise RuntimeError("ObsHistoryBuffer.window() before any observation was appended")
        k = self.steps_per_obs
        t = (self._num_obs - 1) * k  # action step of the latest observation
        first_kept = self._num_obs - len(self._obs)  # observation index of self._obs[0]
        return [self._obs[int(r) // k - first_kept] for r in history_rows(t, self.history_offsets, 0)]

    def reset(self) -> None:
        """Forget the episode (call between episodes)."""
        self._obs.clear()
        self._num_obs = 0
