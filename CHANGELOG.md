# Changelog

All notable changes to this repository are recorded here. The format follows
Keep a Changelog and the project uses semantic versioning.

## Unreleased

### Added

- Python 3.11 package `theoros_models` built with hatchling, with a committed
  uv lock file, ruff, mypy in strict mode, pytest, Hypothesis and per-path
  coverage floors (90 percent, 95 percent for numerics).
- Continuous integration for lint, format, type check, tests with coverage
  gate, repository hygiene and package build.
- Requirements MOD-REQ-009 to MOD-REQ-012, MOD-REQ-020 to MOD-REQ-031,
  MOD-REQ-040 to MOD-REQ-044 and MOD-REQ-220, with verification methods and
  status for every requirement.
- Sensor profile schema with validation of band count, wavelength range,
  order, near-duplicate bands, ground sample distance, reflectance encoding
  and tabulated spectral responses, and a SHA-256 profile fingerprint.
- Spectral response handling: Gaussian band model, resampling of tabulated
  responses and band-equivalent reflectance on a 1 nm grid.
- Catalogue of six profiles: Sentinel-2A MSI, Landsat 8 OLI, Landsat 9 OLI-2,
  PlanetScope SuperDove, a generic 4-band CubeSat and a 5-band test sensor.
  Band values of the real instruments are not yet verified against their
  primary source documents (MOD-REQ-031 open).
- Fixed-point numerics shared with theoros-edge: rounding modes, saturating
  arithmetic, Q31 multipliers, high multiplication, requantisation and affine
  int8 quantisation, specified in docs/NUMERICS.md.
- `theoros sensors list`, `show` and `validate` commands.
- PyTorch dependency with lock file variants for CPU and CUDA 12.6
  (`--extra cpu`, `--extra cu126`); deterministic algorithms in tests.
- D4 group, lift, group, depthwise and invariant convolutions and per-field
  batch normalisation, exactly equivariant to the 8 symmetries of the square
  (MOD-REQ-007, MOD-REQ-062, MOD-REQ-063).
- Spectral hypernetwork stem generating per-band 1 x 1 kernels from centre
  wavelength, FWHM and GSD (MOD-REQ-060, MOD-REQ-061).
- Inverted residual backbone at strides 4, 8 and 16, feature pyramid to
  stride 4, and cloud, land and water, vessel and fire heads (MOD-REQ-011,
  MOD-REQ-064).
- Capability gating with a physical justification for every spectral window
  (MOD-REQ-004, MOD-REQ-065).
- Tiers nano, base and max, and an analytic budget of parameters, MACs, peak
  activation memory and int8 size, checked against the built network
  (MOD-REQ-001, MOD-REQ-002, MOD-REQ-010, MOD-REQ-066, MOD-REQ-067).
- `theoros budget` command; `theoros sensors show` reports capabilities.
- docs/ARCHITECTURE.md.
- Data pipeline: reflectance conversion and normalisation, band resampling,
  tiling and stitching (MOD-REQ-100 to MOD-REQ-102).
- Tile store format with checksums, synthetic scene generator, dataset
  adapters behind cards, and a guard on test splits (MOD-REQ-103 to
  MOD-REQ-106).
- D4-consistent geometric augmentation, spectral jitter and band dropout
  (MOD-REQ-107, MOD-REQ-108).
- Packing of variable band counts and balanced, rank-aware sampling
  (MOD-REQ-109, MOD-REQ-110).
- Masked multi-task losses and vessel heatmap targets (MOD-REQ-120,
  MOD-REQ-121), and an integration test training every tier for one epoch
  of synthetic data on CPU (MOD-REQ-111).
- datasets/README.md, the synthetic dataset card and docs/TRAINING.md.

### Changed

- docs/REQUIREMENTS.md uses the block format of the engineering standards and
  records a verification status for each requirement. MOD-REQ-001 to
  MOD-REQ-008 keep their wording.
- MOD-REQ-010 counts tier sizes on the compiled flight graph at 13 bands, the
  form the nominal sizes and the 1 MiB limit refer to.
- CI runs on every push, installs the CPU build of PyTorch, and checks the
  built wheel with `theoros budget`.
- The dataset card template records version, checksum and adapter.
- The vessel and fire output biases are initialised to a prior probability
  of 0.01.
