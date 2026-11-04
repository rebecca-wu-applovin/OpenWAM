"""Auxiliary package for the Cosmos3-Edge video backbone.

Stateless helpers live here; the backbone class
lives one level up in ``cosmos3/backbone.py``. Importing this package must stay
CPU-CI safe — anything that pulls ``diffusers``/``transformers`` (including
``_vendor/``) is imported lazily inside ``from_pretrained``-reachable code only.
"""
