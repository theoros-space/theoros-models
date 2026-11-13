# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""One training sample: a reflectance tile, its bands and its labels.

Label conventions (MOD-REQ-109):

* ``cloud``: uint8 (H, W), 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow;
* ``land_water``: uint8 (H, W), 0 land, 1 water;
* ``fire``: uint8 (H, W), 0 no active fire, 1 active fire;
* ``vessels``: float32 (K, 4), each row the centre row and column in pixel
  coordinates (pixel ``i`` spans ``i`` to ``i + 1``), length in m and width in m.

In dense labels 255 marks pixels without a valid label. A task whose label
is None is unlabelled for the whole sample and contributes no loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.sensors.schema import MAX_BANDS, MIN_BANDS, Band

IGNORE = 255
CLOUD_CLASSES = 4
DENSE_TASKS = ("cloud", "land_water", "fire")
VESSEL_COLUMNS = 4
_DENSE_MAXIMUM = {"cloud": CLOUD_CLASSES - 1, "land_water": 1, "fire": 1}


@dataclass(frozen=True, eq=False)
class Sample:
    """A tile of TOA reflectance with band metadata and optional labels.

    Attributes:
        image: TOA reflectance, float32 (N, H, W), H equal to W.
        bands: The N bands in image channel order.
        sensor: Name of the sensor profile the tile comes from.
        sample_id: Identifier unique within its dataset.
        cloud: Cloud classes or None.
        land_water: Water mask or None.
        fire: Active fire mask or None.
        vessels: Vessel annotations or None; an empty (0, 4) array means
            labelled with no vessel present.
    """

    image: NDArray[np.float32]
    bands: tuple[Band, ...]
    sensor: str
    sample_id: str
    cloud: NDArray[np.uint8] | None = None
    land_water: NDArray[np.uint8] | None = None
    fire: NDArray[np.uint8] | None = None
    vessels: NDArray[np.float32] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.dtype != np.float32:  # noqa: PLR2004
            raise ValueError("image must be float32 (N, H, W)")
        count, height, width = self.image.shape
        if height != width:
            raise ValueError(f"tiles must be square, got {height}x{width}")
        if count != len(self.bands) or not MIN_BANDS <= count <= MAX_BANDS:
            raise ValueError(f"image has {count} channels for {len(self.bands)} bands")
        for task in DENSE_TASKS:
            label = getattr(self, task)
            if label is None:
                continue
            if label.shape != (height, width) or label.dtype != np.uint8:
                raise ValueError(f"{task} label must be uint8 ({height}, {width})")
            valid = label[label != IGNORE]
            if valid.size and valid.max() > _DENSE_MAXIMUM[task]:
                raise ValueError(f"{task} label has values above {_DENSE_MAXIMUM[task]}")
        if self.vessels is not None and (
            self.vessels.ndim != 2  # noqa: PLR2004
            or self.vessels.shape[1] != VESSEL_COLUMNS
            or self.vessels.dtype != np.float32
        ):
            raise ValueError("vessels must be float32 (K, 4)")

    @property
    def size(self) -> int:
        """Tile edge in pixels."""
        return int(self.image.shape[-1])

    def labelled(self) -> tuple[bool, ...]:
        """Return, in head order, whether each task has a label."""
        return tuple(
            getattr(self, "vessels" if task == "vessel" else task) is not None
            for task in HEAD_NAMES
        )

    def labelled_tasks(self) -> tuple[str, ...]:
        """Return the names of the labelled tasks in head order."""
        return tuple(
            task for task, present in zip(HEAD_NAMES, self.labelled(), strict=True) if present
        )
