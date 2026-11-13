# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for dataset adapters, cards and the tile store (MOD-REQ-103 to 105)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from theoros_models.data.datasets import SyntheticDataset, TileStoreDataset, write_tile_store
from theoros_models.sensors import load_catalogue_profile

ROOT = Path(__file__).resolve().parents[2]
CARD_FIELDS = (
    "Source",
    "Version",
    "License",
    "Commercial use allowed",
    "Sensor and level",
    "Tasks and labels",
    "Checksum",
    "Splits",
    "Adapter",
)


def _profiles() -> dict[str, object]:
    return {name: load_catalogue_profile(name) for name in ("sentinel2_msi", "test_5band")}


def test_cards_complete() -> None:
    cards = sorted((ROOT / "datasets").glob("*.md"))
    cards = [card for card in cards if card.name != "README.md"]
    assert [card.stem for card in cards] == [SyntheticDataset.card]
    for card in cards:
        text = card.read_text(encoding="utf-8")
        for field in CARD_FIELDS:
            line = next(line for line in text.splitlines() if line.startswith(f"| {field} |"))
            assert line.split("|")[2].strip(), field


def test_synthetic_dataset() -> None:
    profiles = list(_profiles().values())
    dataset = SyntheticDataset(profiles, 5, size=32, seed=2, tasks=("land_water", "cloud"))  # type: ignore[arg-type]
    assert len(dataset) == 5
    assert dataset.tasks == ("cloud", "land_water")
    assert dataset.group(1) == ("test_5band", ("cloud", "land_water"))
    assert dataset[1].sensor == "test_5band"
    assert dataset[2].image.shape == (13, 32, 32)
    with pytest.raises(IndexError):
        dataset[5]
    with pytest.raises(ValueError, match="at least one profile"):
        SyntheticDataset([], 1)


def _store(tmp_path: Path) -> tuple[Path, SyntheticDataset]:
    dataset = SyntheticDataset(list(_profiles().values()), 6, size=32, seed=4)  # type: ignore[arg-type]
    splits = ["train", "train", "validation", "test", "train", "validation"]
    root = write_tile_store(
        tmp_path / "store",
        "synthetic-store",
        "synthetic",
        ((dataset[i], split) for i, split in enumerate(splits)),
        _profiles(),  # type: ignore[arg-type]
    )
    return root, dataset


def test_tile_store_roundtrip(tmp_path: Path) -> None:
    root, dataset = _store(tmp_path)
    train = TileStoreDataset(root, "train")
    assert len(train) == 3
    assert train.card == "synthetic"
    assert [row["sample_id"] for row in train.rows] == [dataset[i].sample_id for i in (0, 1, 4)]
    for index, original in zip(range(3), (dataset[0], dataset[1], dataset[4]), strict=True):
        loaded = train[index]
        assert loaded.bands == original.bands
        assert train.group(index) == (original.sensor, original.labelled_tasks())
        expected = np.clip(
            original.image, -0.1 if original.sensor == "sentinel2_msi" else 0.0, None
        )
        np.testing.assert_allclose(loaded.image, expected, atol=0.5e-4 + 1e-7)
        for task in ("cloud", "land_water", "fire", "vessels"):
            np.testing.assert_array_equal(getattr(loaded, task), getattr(original, task))


def test_tile_store_detects_corruption(tmp_path: Path) -> None:
    root, _ = _store(tmp_path)
    store = TileStoreDataset(root, "validation")
    path = root / "samples" / f"{store.rows[0]['sample_id']}.npz"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(OSError, match="checksum mismatch"):
        store[0]
    assert TileStoreDataset(root, "validation", verify=False)[1].sensor == "test_5band"


def test_test_split_guarded(tmp_path: Path) -> None:
    root, _ = _store(tmp_path)
    with pytest.raises(PermissionError, match="final evaluation only"):
        TileStoreDataset(root, "test")
    assert len(TileStoreDataset(root, "test", evaluation=True)) == 1
    with pytest.raises(ValueError, match="unknown split"):
        TileStoreDataset(root, "holdout")


def test_tile_store_rejects_bad_input(tmp_path: Path) -> None:
    dataset = SyntheticDataset([load_catalogue_profile("test_5band")], 1, size=16)
    with pytest.raises(ValueError, match="unknown split"):
        write_tile_store(tmp_path / "a", "a", "synthetic", [(dataset[0], "dev")], _profiles())  # type: ignore[arg-type]
    root, _ = _store(tmp_path)
    meta = yaml.safe_load((root / "dataset.yaml").read_text())
    meta["card"] = ""
    (root / "dataset.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="missing card"):
        TileStoreDataset(root, "train")


def test_tile_store_unlabelled_tasks(tmp_path: Path) -> None:
    profile = load_catalogue_profile("test_5band")
    dataset = SyntheticDataset([profile], 1, size=16, tasks=("cloud",))
    root = write_tile_store(
        tmp_path / "s", "s", "synthetic", [(dataset[0], "train")], {"test_5band": profile}
    )
    loaded = TileStoreDataset(root, "train")[0]
    assert loaded.labelled_tasks() == ("cloud",)
    assert loaded.vessels is None
    assert loaded.fire is None
