"""Frozen, dependency-light Gate-4 base-characterization registry."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Literal


B_VALUES = (2.0, 3.0, 5.0, 8.0)
W_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
Q_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
PARAMETER_CELLS = tuple(product(B_VALUES, W_VALUES, Q_VALUES))
HELDOUT_POLICIES = (
    "forgiving_grudger",
    "delayed_tit_for_tat",
    "probabilistic_defector",
    "copy_with_noise_10%",
)


@dataclass(frozen=True, slots=True)
class Gate4Assignment:
    scenario: str
    suite: str
    partner_policy: str
    policy_split: Literal["heldout", "diagnostic"]
    mode: Literal["dyadic", "group"] = "dyadic"
    switch_to_policy: str | None = None
    switch_direction: str | None = None
    group_size: int = 4


@lru_cache(maxsize=None)
def _indices_for_category(seed_offset: int, category: int) -> tuple[int, ...]:
    result = []
    for index, (b, w, q) in enumerate(PARAMETER_CELLS):
        b_index = B_VALUES.index(b)
        w_index = W_VALUES.index(w)
        q_index = Q_VALUES.index(q)
        if (b_index + w_index + q_index + seed_offset) % 5 == category:
            result.append(index)
    if len(result) != 20:
        raise AssertionError("each Gate-4 Latin category must contain 20 cells")
    return tuple(result)


@lru_cache(maxsize=None)
def _reserved_indices(seed_offset: int, category: int) -> frozenset[int]:
    """Choose five cells balanced over w/q and nearly balanced over b."""

    candidates = _indices_for_category(seed_offset, category)
    preferred_extra_b = (seed_offset + category) % len(B_VALUES)
    feasible = []
    for subset in combinations(candidates, 5):
        cells = [PARAMETER_CELLS[index] for index in subset]
        if {w for _, w, _ in cells} != set(W_VALUES):
            continue
        if {q for _, _, q in cells} != set(Q_VALUES):
            continue
        b_counts = Counter(B_VALUES.index(b) for b, _, _ in cells)
        counts = tuple(b_counts[index] for index in range(len(B_VALUES)))
        feasible.append(
            (
                max(counts) - min(counts),
                -counts[preferred_extra_b],
                sum((count - 1.25) ** 2 for count in counts),
                subset,
            )
        )
    if not feasible:
        raise AssertionError("Gate-4 balanced scenario reservation is infeasible")
    return frozenset(min(feasible)[-1])


@lru_cache(maxsize=None)
def _group_indices(seed_offset: int) -> tuple[int, ...]:
    return tuple(
        sorted(_reserved_indices(seed_offset, 3) | _reserved_indices(seed_offset, 4))
    )


@lru_cache(maxsize=None)
def _exploitability_indices(seed_offset: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            _reserved_indices(seed_offset, 0)
            | _reserved_indices(seed_offset, 1)
            | _reserved_indices(seed_offset, 2)
        )
    )


@lru_cache(maxsize=None)
def _switch_indices(seed_offset: int) -> tuple[int, ...]:
    return tuple(
        index
        for index in _indices_for_category(seed_offset, 4)
        if index not in _reserved_indices(seed_offset, 4)
    )


def gate4_assignment(evaluation_seed: int, cell_index: int) -> Gate4Assignment:
    if evaluation_seed not in range(2101, 2106):
        raise ValueError("Gate-4 requires validation seed 2101–2105")
    if cell_index not in range(len(PARAMETER_CELLS)):
        raise ValueError("Gate-4 cell index must be in 0–99")
    seed_offset = evaluation_seed - 2101
    b, w, q = PARAMETER_CELLS[cell_index]
    category = (
        B_VALUES.index(b) + W_VALUES.index(w) + Q_VALUES.index(q) + seed_offset
    ) % 5
    reserved = cell_index in _reserved_indices(seed_offset, category)
    if reserved and category <= 2:
        rank = _exploitability_indices(seed_offset).index(cell_index)
        partner = "always_defect" if rank % 2 == 0 else "opportunist"
        return Gate4Assignment(
            "diagnostic_exploitability",
            "exploitability",
            partner,
            "diagnostic",
        )
    if reserved:
        rank = _group_indices(seed_offset).index(cell_index)
        partner = HELDOUT_POLICIES[rank % len(HELDOUT_POLICIES)]
        return Gate4Assignment(
            "heldout_group_forecast",
            "forecast",
            partner,
            "heldout",
            mode="group",
            group_size=4 + (rank % 2),
        )
    if category == 0:
        return Gate4Assignment(
            "heldout_forgiving_grudger",
            "nowak",
            "forgiving_grudger",
            "heldout",
        )
    if category == 1:
        return Gate4Assignment(
            "heldout_delayed_tft",
            "nowak",
            "delayed_tit_for_tat",
            "heldout",
        )
    if category == 2:
        return Gate4Assignment(
            "heldout_probabilistic_defector",
            "nowak",
            "probabilistic_defector",
            "heldout",
        )
    if category == 3:
        return Gate4Assignment(
            "heldout_noisy_copy",
            "amtft",
            "copy_with_noise_10%",
            "heldout",
        )
    switch_rank = _switch_indices(seed_offset).index(cell_index)
    if (switch_rank + seed_offset) % 2:
        return Gate4Assignment(
            "diagnostic_switch",
            "switch",
            "always_defect",
            "diagnostic",
            switch_to_policy="tit_for_tat",
            switch_direction="AD_to_TFT",
        )
    return Gate4Assignment(
        "diagnostic_switch",
        "switch",
        "tit_for_tat",
        "diagnostic",
        switch_to_policy="always_defect",
        switch_direction="TFT_to_AD",
    )
