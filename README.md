<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/banners/models-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/banners/models-light.png">
  <img src="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/banners/models-light.png" width="100%" alt="theoros-models">
</picture>

<p><a href="LICENSE"><img src="https://img.shields.io/badge/license-source--available%201.0-FF4B1F?style=flat-square&labelColor=11151C" alt="license: source-available 1.0"></a> <img src="https://img.shields.io/badge/status-pre--release-8B929D?style=flat-square&labelColor=11151C" alt="status: pre-release"> <a href="https://github.com/theoros-space"><img src="https://img.shields.io/badge/part-01%20of%2006-3E7BFA?style=flat-square&labelColor=11151C" alt="part: 01 of 06"></a></p>

> [!NOTE]
> **Status:** Pre-release. The repository is public so partners and reviewers can follow the work. Source code lands here from December 2026.

## Overview

TheorosNet is one network for every Earth observation satellite. It is
trained once on many sensors, then compiled on the ground for each customer's
instrument and flight computer. What flies is a static int8 network with fewer
than one million parameters that fits in 1 MiB.

## Where it fits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/diagrams/flow-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/diagrams/flow-light.png">
  <img src="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/diagrams/flow-light.png" width="100%" alt="How Theoros works: sensor, onboard inference, answers, ground, compiler, signed update">
</picture>

| | Repository | Role |
|:-:|---|---|
| `01` | **theoros-models** | Multi-task model, training and the ground compiler |
| `02` | [theoros-edge](https://github.com/theoros-space/theoros-edge) | Onboard int8 runtime and flight targets |
| `03` | [theoros-delivery](https://github.com/theoros-space/theoros-delivery) | Signed model packages and over-the-air updates |
| `04` | [theoros-ground](https://github.com/theoros-space/theoros-ground) | Customer API, mission dashboard and result tracking |
| `05` | [theoros-bench](https://github.com/theoros-space/theoros-bench) | Open benchmark for onboard Earth observation AI |
| `06` | [theoros-docs](https://github.com/theoros-space/theoros-docs) | Technical documentation and flight documents |

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/diagrams/architecture-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/diagrams/architecture-light.png">
  <img src="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/github/diagrams/architecture-light.png" width="100%" alt="TheorosNet: bands, spectral stem, D4 lift, backbone, neck, four heads; three tiers">
</picture>

| Stage | What it does |
|---|---|
| Spectral stem | Builds first-layer weights from each band's centre wavelength, bandwidth and ground sample distance, so any band set can be read |
| D4 lift | First convolution shared over the 8 rotations and mirrors of the square. In orbit an image has no "up" |
| Backbone | Inverted residual blocks at strides 4, 8 and 16 |
| Neck | Light feature pyramid back to stride 4 for dense outputs |
| Heads | Cloud mask, land and water, vessels, active fire |
| Ground compiler | Trained model plus one sensor profile becomes a static flight graph with calibration folded in |

### Tiers

| Tier | Parameters | Intended hardware |
|---|--:|---|
| `nano` | about 63K | Flight computers without an AI accelerator |
| `base` | about 450K | Flight computers with an AI accelerator |
| `max` | about 890K | Highest accuracy, still under 1 MiB in int8 |

### Guarantees checked on every change

- Parameter count and int8 size stay inside the flight budget.
- The same weights compile for any sensor with 3 to 13 bands between 400 and 2400 nm.
- A head is disabled, and reported, when the sensor lacks the bands its task physically needs.
- The compiled network matches the training network within 1e-4 before quantisation.
- Every release ships a model card with accuracy, size and latency per target.

## Repository layout

Planned structure. Directories appear as the code lands.

```text
src/theoros_models/
  sensors.py         sensor profiles: bands, bandwidths, GSD, calibration
  capabilities.py    which heads a given sensor can support
  arch.py            network definition and its analytic flight budget
  models/            TheorosNet in training form
  compiler.py        training form + sensor profile -> static flight network
  export.py          ONNX export, operator check, manifest for delivery
configs/             one file per training stage
cards/               model and dataset card templates
docs/                requirements and design notes
tests/
```

## Requirements

Every requirement lives in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) with
the prefix `MOD-REQ`. Each one names how it is verified (test, analysis or
inspection), and every pull request that changes behaviour cites the
requirement it serves.

## Contributing

Read the [contributing guide](https://github.com/theoros-space/.github/blob/main/CONTRIBUTING.md)
before opening a pull request. Outside contributions need a signed
[Contributor License Agreement](https://github.com/theoros-space/.github/blob/main/CLA.md).
Report vulnerabilities privately as described in the
[security policy](https://github.com/theoros-space/.github/blob/main/SECURITY.md), never in a public issue.

## License

Source-available under the [Theoros Source-Available License 1.0](LICENSE).
You may read and evaluate this repository. Production use, modification,
redistribution and training competing models need a written license from
Theoros: licensing@theoros.space. See [which license applies where](https://github.com/theoros-space/.github/blob/main/LICENSING.md).

---

<p align="center">
  <a href="https://github.com/theoros-space"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/logo/svg/mark-paper.svg">
    <img src="https://raw.githubusercontent.com/theoros-space/theoros-brand/main/logo/svg/mark-ink.svg" width="28" alt="Theoros">
  </picture></a>
  <br>
  <sub>Theoros &nbsp;·&nbsp; Onboard AI for Earth observation &nbsp;·&nbsp; Baku and Helsinki &nbsp;·&nbsp; <a href="https://theoros.space">theoros.space</a></sub>
</p>
