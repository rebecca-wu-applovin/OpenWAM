"""Deploy self-containment artifacts for cosmos3 (CPU).

save_deploy_assets copies the tokenizer's structural files from wherever the backbone read its
tokenizer (bundle or source checkpoint) into ``<ckpt>/text_tokenizer/``, where from_checkpoint_dir
reads them; it never touches the config (the yaml dit / vae blocks are the reconstruction spec).
"""

import json

from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone


def _fake_tokenizer_dir(tmp_path):
    """Returns: a text_tokenizer/ dir with the four structural files."""
    tok = tmp_path / "bundle" / "text_tokenizer"
    tok.mkdir(parents=True)
    (tok / "tokenizer.json").write_text(json.dumps({"version": "fake"}))
    (tok / "tokenizer_config.json").write_text("{}")
    (tok / "special_tokens_map.json").write_text("{}")
    (tok / "chat_template.jinja").write_text("{{ messages }}")
    return tok


def _backbone_reading(tok_dir):
    """Returns: a bare Cosmos3EdgeVideoBackbone whose tokenizer was read from ``tok_dir``."""
    vb = Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)
    object.__setattr__(vb, "_tokenizer_dir", str(tok_dir))
    return vb


def test_save_deploy_assets_copies_tokenizer_files(tmp_path):
    vb = _backbone_reading(_fake_tokenizer_dir(tmp_path))
    out = tmp_path / "ckpt"
    vb.save_deploy_assets(str(out))
    copied = sorted(p.name for p in (out / "text_tokenizer").iterdir())
    assert copied == ["chat_template.jinja", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    # Only the tokenizer dir: nothing else is written.
    assert sorted(p.name for p in out.iterdir()) == ["text_tokenizer"]


def test_save_deploy_assets_repairs_partial_state(tmp_path):
    vb = _backbone_reading(_fake_tokenizer_dir(tmp_path))
    out = tmp_path / "ckpt"
    vb.save_deploy_assets(str(out))
    copied = sorted(p.name for p in (out / "text_tokenizer").iterdir())
    # Idempotent: a second call leaves the populated dir untouched.
    vb.save_deploy_assets(str(out))
    assert sorted(p.name for p in (out / "text_tokenizer").iterdir()) == copied
    # Repair: a partial copy (interrupted save) is completed, not frozen.
    (out / "text_tokenizer" / "tokenizer.json").unlink()
    vb.save_deploy_assets(str(out))
    assert (out / "text_tokenizer" / "tokenizer.json").exists()


def test_checkpoint_built_backbone_relays_its_assets(tmp_path):
    """A backbone built from a checkpoint (finetune) copies the checkpoint's own files into the new run dir."""
    first = tmp_path / "run1"
    _backbone_reading(_fake_tokenizer_dir(tmp_path)).save_deploy_assets(str(first))
    second = tmp_path / "run2"
    _backbone_reading(first / "text_tokenizer").save_deploy_assets(str(second))
    assert (second / "text_tokenizer" / "tokenizer.json").read_text() == json.dumps({"version": "fake"})
