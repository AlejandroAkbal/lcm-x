"""R2 host observer entry: imported by the host python at start-up because the R2 runner puts this directory on
PYTHONPATH of the host process it spawns. It does nothing unless REL_OBSERVER_DIR is set (see rel_observer.py)."""
import os

if os.environ.get("REL_OBSERVER_DIR"):
    import rel_observer

    rel_observer.install()
