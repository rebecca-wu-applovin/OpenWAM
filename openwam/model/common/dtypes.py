"""``DTYPES``: ``training.mixed_precision`` -> the torch dtype the weights run in, for training and deploy alike.

Neutral package (``openwam.model.common``) so the trainer, the config checks and deploy read one table.
"""

from __future__ import annotations

import torch

# The only legal training.mixed_precision values; resolve_config rejects any other key.
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
