<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/logo/svg/mark-paper.svg">
  <img src="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/logo/svg/mark-ink.svg" width="44" align="right" alt="Theoros">
</picture>

# theoros-models

**01  Multi-task model, training and ground compiler.** Part of the [Theoros](https://github.com/theoros-space) onboard AI product.

[![License: Theoros Source-Available 1.0](https://img.shields.io/badge/license-source--available-FF4B1F)](LICENSE)

> **Status: pre-release.** This repository is public so partners can follow our work. Source code lands here from December 2026.

## What it is

TheorosNet is one network for every Earth observation satellite. It is
trained once on many sensors and compiled on the ground for each customer's
instrument and processor. What flies is a static int8 network under one
million parameters.

| Part | Role |
|---|---|
| Spectral stem | Reads any band set from each band's physics: centre wavelength, bandwidth, ground sample distance |
| D4 lift | First convolution shared over the 8 rotations and mirrors of the square, so the image has no "up" |
| Backbone | Inverted residual blocks at strides 4, 8 and 16 |
| Neck | Light feature pyramid back to stride 4 |
| Heads | Cloud and land/water segmentation, vessel detection, active fire |
| Ground compiler | Turns the trained model plus one sensor profile into a static flight graph |

Three tiers share the design: **nano** (about 63K parameters) for computers
without an AI accelerator, **base** (about 450K) and **max** (about 890K) for
AI processors.

## Planned layout

```
src/theoros_models/
  sensors.py         sensor profiles: bands, bandwidths, GSD, calibration
  capabilities.py    which heads are valid for a given sensor
  arch.py            the network definition and its analytic flight budget
  models/            TheorosNet in its training form
  compiler.py        training form + sensor -> static flight network
  export.py          ONNX export, operator check, manifest for delivery
configs/             one file per training stage
cards/               model and dataset cards
docs/                requirements and design notes
tests/
```

## Requirements

Requirements are listed in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) with
the prefix `MOD-REQ`. Every requirement is verified by a test, an
analysis or an inspection, and referenced in the pull request that changes it.

## Contributing

See the organisation [contributing guide](https://github.com/theoros-space/.github/blob/main/CONTRIBUTING.md).
Outside contributions need a signed [CLA](https://github.com/theoros-space/.github/blob/main/CLA.md).

## License

Source-available under the [Theoros Source-Available License 1.0](LICENSE).
You may read and evaluate it; any use, copy or deployment needs a written
license from Theoros. Contact licensing@theoros.space.
