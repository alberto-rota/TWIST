"""Sequence-level train / validation splitting.

Splits happen at the *sequence* (video) level, never within a sequence, so a
clip never leaks between train and val. Two modes:

* **explicit** -- pass ``train_sequences`` and/or ``val_sequences`` lists.
* **fractional** -- a deterministic random split controlled by ``val_fraction``
  and ``seed`` (ergonomic for the 270 numbered CT3Kubric sequences).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def split_sequences(
    sequences: Sequence[str],
    val_fraction: float = 0.1,
    seed: int = 42,
    train_sequences: Optional[Sequence[str]] = None,
    val_sequences: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Return ``(train, val)`` sequence-name lists.

    Resolution order:

    1. If ``val_sequences`` and/or ``train_sequences`` are given, they are used
       verbatim (intersected with what actually exists). When only one side is
       given, the other becomes "everything else".
    2. Otherwise a deterministic shuffle (by ``seed``) assigns the last
       ``val_fraction`` of sequences to validation.

    Args:
        sequences: all available sequence names (any order).
        val_fraction: fraction held out for validation in fractional mode.
        seed: RNG seed for the deterministic shuffle.
        train_sequences: explicit training sequence names (optional).
        val_sequences: explicit validation sequence names (optional).
    """
    allset = list(dict.fromkeys(sequences))  # de-dup, keep order
    present = set(allset)

    explicit_val = [s for s in (val_sequences or []) if s in present]
    explicit_train = [s for s in (train_sequences or []) if s in present]

    if explicit_val or explicit_train:
        if explicit_train and not explicit_val:
            train = explicit_train
            val = [s for s in allset if s not in set(explicit_train)]
        elif explicit_val and not explicit_train:
            val = explicit_val
            train = [s for s in allset if s not in set(explicit_val)]
        else:
            train, val = explicit_train, explicit_val
        return train, val

    # Fractional, deterministic split.
    if not (0.0 <= val_fraction <= 1.0):
        raise ValueError(f"val_fraction must be in [0, 1], got {val_fraction}")
    # val_fraction=1.0: all sequences to val, nothing to train (eval-only datasets).
    if val_fraction == 1.0:
        return [], list(allset)
    order = np.array(sorted(allset))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    n_val = int(round(len(order) * val_fraction))
    # Guarantee a non-empty val (and a non-empty train) whenever a fraction is
    # asked for and there are >= 2 sequences -- important for tiny smoke runs.
    if val_fraction > 0 and len(order) >= 2:
        n_val = min(max(n_val, 1), len(order) - 1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train = sorted(order[train_idx].tolist())
    val = sorted(order[val_idx].tolist())
    return train, val
