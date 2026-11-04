"""``gradient_checkpoint_forward``: one block call, gradient-checkpointed (optionally with CPU offload) or plain.

Model-level (neutral) so the video and action backbones share it without one importing the other's package.
"""

import torch


def gradient_checkpoint_forward(model, use_gradient_checkpointing, use_gradient_checkpointing_offload, *args):
    """Run ``model(*args)`` under the requested activation-memory policy.

    Args:
        model: callable block (nn.Module or function).
        use_gradient_checkpointing: bool, recompute activations in backward (``torch.utils.checkpoint``).
        use_gradient_checkpointing_offload: bool, checkpoint with saved tensors offloaded to CPU (wins over the
            plain flag).
        *args: the block's positional inputs.

    Returns:
        the block's output, numerically identical under every policy.
    """
    if use_gradient_checkpointing_offload:
        # Checkpointed and the remaining saved tensors parked on the CPU.
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(model, *args, use_reentrant=False)
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(model, *args, use_reentrant=False)
    return model(*args)
