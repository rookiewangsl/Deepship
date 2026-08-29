"""Helpers for deterministic repeated vessel-disjoint DeepShip protocols."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from src.data.deepship_protocols import compile_protocol


def make_repeat_experiment_config(
    base_config: Mapping[str, object],
    *,
    split_seed: int,
    experiment_prefix: str = "macnna_global_l20_repeats_v1",
) -> dict[str, object]:
    """Clone the frozen isolation config for one new vessel split.

    The 3-second segment budget remains unchanged because it defines eligible
    anchors in the frozen manifest.  L20 waveform loading is a documented
    training override shared by G0, G0-C, and G1.
    """

    config = deepcopy(dict(base_config))
    split = config.get("split")
    if not isinstance(split, dict):
        raise TypeError("Base experiment split section must be an object")
    config["experiment_id"] = f"{experiment_prefix}_split{int(split_seed)}"
    config["description"] = (
        "Repeated vessel-name-disjoint DeepShip partition for the L20 global-attention study."
    )
    split["split_seed"] = int(split_seed)
    return config


def compile_repeat_vessel_split(
    base_config: Mapping[str, object],
    inventory_rows: list[dict[str, object]],
    identity_rows: list[dict[str, object]],
    exclusion_rows: list[dict[str, object]],
    *,
    split_seed: int,
    source_inventory_sha256: str,
    source_identity_sha256: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Compile one deterministic repeat and return config plus protocol outputs."""

    config = make_repeat_experiment_config(base_config, split_seed=split_seed)
    manifest, recordings, groups, report = compile_protocol(
        "vessel_name_disjoint",
        config,
        inventory_rows,
        identity_rows,
        exclusion_rows,
        source_inventory_sha256=source_inventory_sha256,
        source_identity_sha256=source_identity_sha256,
    )
    return config, manifest, recordings, groups, report
