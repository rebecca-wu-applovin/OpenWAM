"""Offline stats precompute tools for dataloaders.

``yam_sharpa_stats_computation``: scans the yam_sharpa sources once and writes the pooled normalization stats the
dataset requires at train time (also built on first use by ``MultiYamSharpaDataset.from_config``).

Run it as ``python -m openwam.dataloader.utils.stats_computation.yam_sharpa_stats_computation``.
"""
