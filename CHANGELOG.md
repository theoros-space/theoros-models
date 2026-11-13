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
- Fixed-point numerics shared with theoros-edge: rounding modes, saturating
  arithmetic, Q31 multipliers, high multiplication, requantisation and affine
  int8 quantisation, specified in docs/NUMERICS.md.
- `theoros sensors list`, `show` and `validate` commands.

### Changed

- docs/REQUIREMENTS.md uses the block format of the engineering standards and
  records a verification status for each requirement. MOD-REQ-001 to
  MOD-REQ-008 keep their wording.
