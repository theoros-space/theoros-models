# theoros-models requirements

Every requirement has an identifier, a statement and a verification method
(test, analysis or inspection). A test reference names the test that proves
it. Status is `verified` when that verification exists and passes on `main`,
and `open` while the verifying work is still to land, optionally followed by
the reason; an open requirement is never reported as met.

Identifier ranges group the requirements by component:

| Range | Component |
|---|---|
| 001 to 019 | System and flight budget |
| 020 to 039 | Sensor profiles and spectral response |
| 040 to 059 | Fixed-point numerics |
| 060 to 099 | Architecture, capability gating and budget |
| 100 to 119 | Data pipeline |
| 120 to 139 | Losses and training |
| 220 to 239 | Command-line interface |

Status: baseline, November 2026. Changes need approval from the CTO.

## System and flight budget

```text
MOD-REQ-001  A flight model shall have fewer than 1,000,000 parameters after
             BatchNorm folding and D4 expansion.
             Verification: analysis, CI (tests/test_budget.py::test_flight_limits)
             Status: verified

MOD-REQ-002  Its int8 weights and int32 biases shall fit in 1 MiB.
             Verification: analysis, CI (tests/test_budget.py::test_flight_limits)
             Status: verified

MOD-REQ-003  The same trained weights shall compile for any sensor with 3 to
             13 bands between 400 and 2400 nm.
             Verification: test
             Status: open

MOD-REQ-004  A head shall be disabled when the sensor lacks the spectral
             bands its task physically needs.
             Verification: test (tests/test_capabilities.py,
             tests/test_model.py::test_gating_skips_unsupported_heads)
             Status: verified

MOD-REQ-005  The compiled network shall use only the operators in the flight
             operator set.
             Verification: test
             Status: open

MOD-REQ-006  The compiled network shall match the training network within
             1e-4 before quantisation.
             Verification: test
             Status: open

MOD-REQ-007  The lift layer shall be exactly equivariant to the 8 symmetries
             of the square.
             Verification: test (tests/layers/test_d4.py::test_lift_exact_on_integers)
             Status: verified

MOD-REQ-008  Every released model shall ship a model card with accuracy,
             size and latency per target.
             Verification: inspection
             Status: open

MOD-REQ-009  The model shall take square tiles of 256 x 256 pixels of L1C
             top-of-atmosphere reflectance as input.
             Verification: test
             Status: open

MOD-REQ-010  The flight graph of each tier, compiled for 13 bands with all
             heads, shall have a parameter count within 5 percent of its
             nominal size: nano 63,000, base 450,000 and max 890,000.
             Verification: analysis, CI (tests/test_budget.py::test_tier_sizes)
             Status: verified

MOD-REQ-011  The backbone shall produce features at strides 4, 8 and 16,
             and the neck shall fuse them back to stride 4 for the dense
             heads cloud mask, land and water, vessels and active fire.
             Verification: test (tests/test_model.py::test_feature_strides,
             tests/test_model.py::test_output_shapes)
             Status: verified

MOD-REQ-012  The flight graph shall be exported as ONNX opset 17 with static
             shapes only.
             Verification: test
             Status: open
```

## Sensor profiles and spectral response

```text
MOD-REQ-020  A sensor profile shall be a YAML document validated against the
             schema in sensors/schema.py. Unknown fields, missing fields and
             values of the wrong type shall be rejected with an error that
             names the offending field.
             Verification: test (tests/sensors/test_schema.py::test_unknown_field_rejected,
             tests/sensors/test_schema.py::test_missing_field_rejected)
             Status: verified

MOD-REQ-021  A sensor profile shall have 3 to 13 bands.
             Verification: test (tests/property/test_profile_properties.py::test_band_count_bounds)
             Status: verified

MOD-REQ-022  Each band shall have a positive full width at half maximum
             (FWHM), and its half-power interval, centre minus FWHM/2 to
             centre plus FWHM/2, shall lie within 400 to 2400 nm.
             Verification: test (tests/property/test_profile_properties.py::test_band_outside_range_rejected)
             Status: verified

MOD-REQ-023  Bands shall be listed in strictly increasing order of centre
             wavelength, and band names shall be unique within a profile.
             Verification: test (tests/property/test_profile_properties.py::test_unsorted_bands_rejected,
             tests/sensors/test_schema.py::test_duplicate_band_name_rejected)
             Status: verified

MOD-REQ-024  No two bands shall be near-duplicates: the intersection over
             union of their half-power intervals shall not exceed 0.5.
             Verification: test (tests/property/test_profile_properties.py::test_near_duplicate_rejected)
             Status: verified

MOD-REQ-025  Each band shall have a ground sample distance (GSD) between
             0.1 m and 1000 m.
             Verification: test (tests/property/test_profile_properties.py::test_gsd_out_of_range_rejected)
             Status: verified

MOD-REQ-026  A band may carry a tabulated spectral response function (SRF).
             A tabulated SRF shall have at least 3 samples at strictly
             increasing wavelengths within 300 to 2600 nm, non-negative
             responses with a positive peak, a response-weighted centroid
             within 0.25 FWHM of the declared centre, and a measured FWHM
             within 25 percent of the declared FWHM. These tolerances catch
             unit errors and mismatched bands; they are not a calibration
             accuracy claim.
             Verification: test (tests/sensors/test_srf.py::test_inconsistent_srf_rejected)
             Status: verified

MOD-REQ-027  A band without a tabulated SRF shall be modelled by a Gaussian
             response with the declared centre and FWHM.
             Verification: test (tests/sensors/test_srf.py::test_gaussian_measured_fwhm)
             Status: verified

MOD-REQ-028  Band-equivalent reflectance shall be the response-weighted mean
             of a spectrum on a 1 nm grid from 350 to 2500 nm. It shall map
             a constant spectrum to that constant and stay within the
             minimum and maximum of the spectrum.
             Verification: test (tests/property/test_srf_properties.py)
             Status: verified

MOD-REQ-029  Each profile shall have a SHA-256 fingerprint over its
             flight-relevant content (name, platform, instrument, encoding
             and bands). The fingerprint shall not depend on key order,
             formatting, description or reference, and shall change when
             any flight-relevant value changes.
             Verification: test (tests/property/test_profile_properties.py::test_fingerprint_sensitivity,
             tests/sensors/test_schema.py::test_fingerprint_ignores_formatting)
             Status: verified

MOD-REQ-030  A profile shall declare the affine encoding from stored pixel
             values to top-of-atmosphere reflectance, reflectance equals
             scale times value plus offset, with a finite positive scale and
             a finite offset.
             Verification: test (tests/sensors/test_schema.py::test_encoding_validation)
             Status: verified

MOD-REQ-031  The package shall ship a validated catalogue with the profiles
             sentinel2_msi, landsat8_oli, landsat9_oli2,
             planetscope_superdove, generic_cubesat_4band and test_5band.
             Each real instrument profile shall cite the title and version
             of its primary source document, and its band values shall
             match that document.
             Verification: test (tests/sensors/test_catalogue.py) for schema
             validity and citation; inspection against the primary source
             for the band values
             Status: open: values not yet verified against the primary source
```

## Architecture, capability gating and budget

The design behind these requirements is in [ARCHITECTURE.md](ARCHITECTURE.md).

```text
MOD-REQ-060  The spectral stem shall generate one 1 x 1 kernel per band and
             a bias from the centre wavelength, FWHM and GSD of every band
             of the set. Its output shall not depend on the order of the
             bands or on padded band slots, and shall equal a plain 1 x 1
             convolution with the generated weights.
             Verification: test (tests/layers/test_stem.py)
             Status: verified

MOD-REQ-061  The stem shall accept every valid band set of 3 to 13 bands
             between 400 and 2400 nm and produce finite outputs.
             Verification: test (tests/property/test_stem_properties.py)
             Status: verified

MOD-REQ-062  The D4 group action on trivial and regular feature maps shall
             be a representation: the identity acts trivially and acting
             with h then g equals acting with g h.
             Verification: test (tests/layers/test_d4.py::test_regular_action_is_representation)
             Status: verified

MOD-REQ-063  Every layer and the whole network shall be equivariant to D4:
             transforming the input by a symmetry of the square transforms
             every dense output by the same symmetry. The deviation shall
             be at most 1e-9 in float64 and 1e-4 relative to the output
             magnitude in float32, for square tiles whose side is a
             multiple of 16.
             Verification: test (tests/test_model.py::test_network_equivariance,
             tests/layers/test_d4.py, tests/layers/test_blocks.py)
             Status: verified

MOD-REQ-064  Each head shall produce trivial output maps at stride 4 with the
             channels: cloud 4 class logits (clear, thick cloud, thin cloud,
             cloud shadow), land and water 1 water logit, vessel 3 channels
             (centre logit, log length in m, log width in m), active fire
             1 logit.
             Verification: test (tests/test_model.py::test_output_shapes)
             Status: verified

MOD-REQ-065  Capability gating shall follow the window table of
             capabilities.py and docs/ARCHITECTURE.md, and every disabled
             head shall be reported with each missing need and its physical
             reason.
             Verification: test (tests/test_capabilities.py), inspection
             Status: verified

MOD-REQ-066  The budget shall be computed analytically from the tier
             definition. Its training parameter count shall equal the
             parameters of the built network, and its flight weight and
             bias counts shall equal the expanded tensors of that network.
             Verification: test (tests/test_budget.py::test_analytic_matches_module)
             Status: verified

MOD-REQ-067  The budget shall report multiply-accumulates per tile and peak
             int8 activation memory for 256 x 256 tiles, and the command
             theoros budget shall print the flight budget of every tier and
             exit non-zero if any tier breaks MOD-REQ-001, MOD-REQ-002 or
             MOD-REQ-010.
             Verification: test (tests/test_budget.py, tests/test_cli.py::test_budget)
             Status: verified
```

## Data pipeline

```text
MOD-REQ-100  Tiling shall cover every pixel of an image with square windows
             of the tile size that stay inside the image, padding images
             smaller than a tile, and stitching shall average overlapping
             tile predictions back into the image.
             Verification: test (tests/data/test_tiles.py)
             Status: verified

MOD-REQ-101  Stored pixel values shall be converted to TOA reflectance with
             the profile encoding, and the model input shall be
             (clip(reflectance, 0, 1.5) - 0.2) / 0.2.
             Verification: test (tests/data/test_tiles.py::test_reflectance_and_normalisation)
             Status: verified

MOD-REQ-102  Bands of different GSD shall be brought to a common grid by
             nearest-neighbour upsampling or block-mean downsampling by
             integer factors; downsampling shall preserve the mean.
             Verification: test (tests/data/test_tiles.py::test_band_resampling)
             Status: verified

MOD-REQ-103  Every dataset adapter shall name a dataset card in datasets/
             that records source, version, license, commercial use,
             checksum and split definition. Datasets shall never be stored
             in git.
             Verification: test (tests/data/test_datasets.py::test_cards_complete), inspection
             Status: verified

MOD-REQ-104  The tile store shall hold stored values as uint16 through the
             profile encoding, labels, the sensor profiles and a SHA-256 per
             sample file; reading shall reproduce the written samples within
             half an encoding step and reject a file whose checksum differs.
             Verification: test (tests/data/test_datasets.py::test_tile_store_roundtrip,
             tests/data/test_datasets.py::test_tile_store_detects_corruption)
             Status: verified

MOD-REQ-105  A test split shall be readable only when evaluation is
             requested explicitly, so that training and model selection
             cannot use it.
             Verification: test (tests/data/test_datasets.py::test_test_split_guarded)
             Status: verified

MOD-REQ-106  A synthetic scene generator shall render labelled multi-sensor
             tiles for any valid band set, deterministically from a seed,
             with labels consistent with the rendered materials, so that CI
             never downloads data.
             Verification: test (tests/data/test_synthetic.py)
             Status: verified

MOD-REQ-107  Geometric augmentation shall apply the same symmetry of the
             square to the image, every dense label and the vessel centres,
             with the convention of layers/d4.py.
             Verification: test (tests/data/test_augment.py::test_symmetry_matches_model_convention,
             tests/data/test_augment.py::test_vessel_points_follow_pixels)
             Status: verified

MOD-REQ-108  Spectral augmentation shall perturb per-band gain and offset
             and add pixel noise; band dropout shall keep 3 or more bands in
             their original order with matching metadata.
             Verification: test (tests/data/test_augment.py)
             Status: verified

MOD-REQ-109  Samples with different band counts shall be packed into one
             batch with a band mask, unlabelled dense pixels marked 255, a
             per-sample label presence mask, and padded vessel lists.
             Verification: test (tests/data/test_packing.py)
             Status: verified

MOD-REQ-110  The sampler shall draw every group of sensor and labelled tasks
             with equal probability unless weights are given, depend only on
             seed and epoch, and give distributed ranks disjoint shares of
             the same draw sequence.
             Verification: test (tests/data/test_sampler.py)
             Status: verified

MOD-REQ-111  One epoch of synthetic multi-sensor data shall train every tier
             on CPU with finite losses and updates to every parameter.
             Verification: test (tests/integration/test_epoch.py)
             Status: verified
```

## Losses and training

```text
MOD-REQ-120  Each head loss shall average over the samples that carry its
             label and whose band set supports the head; other samples shall
             contribute zero loss and zero gradient through that head, and
             pixels marked 255 shall be ignored.
             Verification: test (tests/test_losses.py::test_masking)
             Status: verified

MOD-REQ-121  Vessel targets shall be Gaussian centre heatmaps at stride 4
             with log length and log width at the centre pixels, and the
             target of a transformed tile shall be the transformed target.
             Verification: test (tests/test_losses.py::test_vessel_targets_equivariant)
             Status: verified
```

## Fixed-point numerics

The definitions referenced below are in [NUMERICS.md](NUMERICS.md). They are
the contract that theoros-edge implements bit-exactly.

```text
MOD-REQ-040  The numerics module shall provide the rounding modes half to
             even, half away from zero, half up, toward zero and floor, and
             apply each exactly to float64 values and to integer division
             by powers of two.
             Verification: test (tests/test_numerics.py::test_round_half_integers_exhaustive,
             tests/test_numerics.py::test_rounding_shift_exhaustive)
             Status: verified

MOD-REQ-041  Saturation to signed 8, 16 and 32 bit ranges, and saturating
             addition and subtraction at those widths, shall equal the
             exact result clamped to the range for every input.
             Verification: test (tests/test_numerics.py::test_saturating_add_sub_int8_exhaustive)
             Status: verified

MOD-REQ-042  A positive real multiplier from 2^-31 to 2^29 shall be
             represented as a Q31 mantissa from 2^30 to 2^31 - 1 and a
             shift from -31 to 30, with relative error at most 2^-31.
             Verification: test (tests/property/test_numerics_properties.py::test_quantize_multiplier_error)
             Status: verified

MOD-REQ-043  Saturating rounding doubling high multiplication and
             requantisation of int32 accumulators to int8 shall follow the
             definitions in NUMERICS.md exactly, and requantised values
             shall differ by at most 1 from the correctly rounded real
             result.
             Verification: test (tests/test_numerics.py::test_srdhm_matches_reference,
             tests/property/test_numerics_properties.py::test_requantize_within_one)
             Status: verified

MOD-REQ-044  Affine int8 quantisation shall support per-tensor and
             per-channel scales and zero points, saturate values outside
             the representable range, and reconstruct in-range values
             within half a quantisation step.
             Verification: test (tests/property/test_numerics_properties.py::test_quantize_roundtrip)
             Status: verified
```

## Command-line interface

```text
MOD-REQ-220  The theoros command shall list the catalogue, show a profile
             with its fingerprint, and validate profile files, exiting with
             a non-zero status and the validation errors when a profile is
             invalid.
             Verification: test (tests/test_cli.py)
             Status: verified
```
