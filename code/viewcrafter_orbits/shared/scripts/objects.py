"""The one place the object list lives.

Experiments 2 through 5 each used to carry their own hardcoded copy of the three
object names and their assumed reference elevations. Adding a fourth object meant
editing six files and hoping none were missed, so the list moved here and every
script reads it.

`elevation` is ViewCrafter's estimate of how far above the object the *reference
camera* sits; it sets the tilt of the orbit plane and is a per-object property of
the capture, not a tuning knob.
"""
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "objects.json"


def load(only=None):
    """[(name, elevation), ...] in file order, optionally filtered by name."""
    rows = json.loads(CONFIG.read_text())
    return [(r["name"], float(r["elevation"])) for r in rows
            if only is None or r["name"] in only]


def names(only=None):
    return [n for n, _ in load(only)]


def captions():
    return {r["name"]: r.get("caption", "") for r in json.loads(CONFIG.read_text())}
