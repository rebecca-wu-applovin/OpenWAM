"""Key-by-key comparison of two run configs (finetune / resume check a launch against a saved config.yaml)."""

# Printed for the side of a difference that lacks the key.
ABSENT = "<absent>"
# Cap on one value's printed length: long lists (e.g. the cosmos3 VAE latents_mean) would bury the changed keys.
_VALUE_MAXLEN = 110


def check_same_cfg(saved: dict, live: dict, ignore: tuple[str, ...], context: str) -> None:
    """Raise if two configs differ on any key outside ``ignore``, naming every differing key.

    Args:
        saved: dict, plain container (``OmegaConf.to_container``) of the saved config.yaml (or a slice of it).
        live: dict, the same for this launch's config; the caller picks the slice and whether to resolve.
        ignore: tuple of dotted keys; a key equal to one of them, or below it, is not compared.
        context: str, the start of the error message (what is compared and why it must match).

    Returns:
        None; raises ValueError ``"<context> on N key(s):"`` + one ``  <key>: saved=<v> launch=<v>`` line per key.
        Lists are compared as one value; a side that lacks the key prints ``'<absent>'``.
    """
    # Flatten both sides to {dotted.key: leaf}; non-empty dicts are walked, everything else is a leaf.
    flat = []
    for tree in (saved, live):
        leaves, stack = {}, [("", tree or {})]
        while stack:
            prefix, node = stack.pop()
            for key, val in node.items():
                if isinstance(val, dict) and val:
                    stack.append((f"{prefix}{key}.", val))
                else:
                    leaves[f"{prefix}{key}"] = val
        flat.append(leaves)

    lines = []
    for key in sorted(flat[0].keys() | flat[1].keys()):
        if any(key == p or key.startswith(p + ".") for p in ignore):
            continue
        pair = [side.get(key, ABSENT) for side in flat]
        if pair[0] != pair[1]:
            # repr each side, cut to _VALUE_MAXLEN characters.
            s, v = (r if len(r) <= _VALUE_MAXLEN else r[: _VALUE_MAXLEN - 3] + "..." for r in map(repr, pair))
            lines.append(f"  {key}: saved={s} launch={v}")
    if lines:
        raise ValueError(f"{context} on {len(lines)} key(s):\n" + "\n".join(lines))
