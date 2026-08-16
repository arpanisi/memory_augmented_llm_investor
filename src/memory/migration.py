"""
Layer migration mechanics for memory promotion and demotion.
Evaluates rules on a single snapshot pass to ensure deterministic single-step movements.
"""
from config.settings import MIGRATION_SHORT_MID, MIGRATION_MID_LONG
from src.memory.store import AssetMemoryStore, MemoryRecord

def run_layer_migration(store: AssetMemoryStore):
    """
    Runs the daily memory flow migration pass across short, mid, and long layers.
    Reflections layer is excluded from migration.
    """
    # 1. Take a snapshot of memory states at the start of the pass
    snapshot_short = list(store.layers["short"])
    snapshot_mid = list(store.layers["mid"])
    snapshot_long = list(store.layers["long"])

    new_short = []
    new_mid = []
    new_long = []

    # 2. Process short layer
    for r in snapshot_short:
        if r.importance >= MIGRATION_SHORT_MID:
            # Promote short -> mid: recency resets to 1.0 (delta = 0)
            r.layer = "mid"
            r.delta = 0
            new_mid.append(r)
        else:
            new_short.append(r)

    # 3. Process mid layer
    for r in snapshot_mid:
        if r.importance >= MIGRATION_MID_LONG:
            # Promote mid -> long: recency resets to 1.0 (delta = 0)
            r.layer = "long"
            r.delta = 0
            new_long.append(r)
        elif r.importance < MIGRATION_SHORT_MID:
            # Demote mid -> short: recency carries over (delta NOT reset)
            r.layer = "short"
            new_short.append(r)
        else:
            new_mid.append(r)

    # 4. Process long layer
    for r in snapshot_long:
        if r.importance < MIGRATION_MID_LONG:
            # Demote long -> mid: recency carries over (delta NOT reset)
            r.layer = "mid"
            new_mid.append(r)
        else:
            new_long.append(r)

    # Update store layers
    store.layers["short"] = new_short
    store.layers["mid"] = new_mid
    store.layers["long"] = new_long
