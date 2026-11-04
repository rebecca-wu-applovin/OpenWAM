"""Auxiliary package for the Cosmos3-Edge video backbone.

Stateless helpers live here (``dit.py``, ``text.py``); the backbone class
lives in ``cosmos3/backbone.py`` of this package. Importing this package must stay
CPU-CI safe — anything that pulls ``diffusers``/``transformers`` (including
``_vendor/``) is imported lazily inside ``from_pretrained``-reachable code only.
"""
