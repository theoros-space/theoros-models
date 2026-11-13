# Datasets

Datasets never enter git. Each dataset used by this repository has a card
here, `<name>.md`, filled from `cards/DATASET_CARD_TEMPLATE.md`, with source,
version, license, commercial use, checksum and split definition
(MOD-REQ-103). A license is verified and recorded before a dataset is used.
Test splits are used only for final evaluation, never for training or model
selection (MOD-REQ-105).

| Card | Status |
|---|---|
| [synthetic](synthetic.md) | in use for tests and pipeline checks |

## Tile store format

External datasets are converted once into a tile store, which
`TileStoreDataset` reads (MOD-REQ-104). Format version 1:

```text
<root>/
  dataset.yaml          format: 1, name, card (name of the card in datasets/)
  index.csv             sample_id, split, sensor, tasks, sha256
  profiles/<sensor>.yaml  sensor profile of every sensor in the index
  samples/<sample_id>.npz
```

- `split` is `train`, `validation` or `test`.
- `tasks` lists the labelled tasks separated by `|`, in head order.
- `sha256` is the digest of the sample file; reading verifies it by default.

Each sample file is a compressed numpy archive:

| Array | Type and shape | Meaning |
|---|---|---|
| `values` | uint16 (N, H, W) | stored values; reflectance is `scale * value + offset` from the profile encoding |
| `cloud` | uint8 (H, W), optional | 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow, 255 unlabelled |
| `land_water` | uint8 (H, W), optional | 0 land, 1 water, 255 unlabelled |
| `fire` | uint8 (H, W), optional | 0 no fire, 1 active fire, 255 unlabelled |
| `vessels` | float32 (K, 4), optional | centre row, centre column (pixels), length (m), width (m) |

Channels are in the band order of the sensor profile, resampled to a common
grid with `upsample_band` or `downsample_band` before writing. A missing
array means the task is unlabelled for that sample.
