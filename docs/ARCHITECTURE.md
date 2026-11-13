# TheorosNet architecture

TheorosNet is one network for many sensors. It is trained once in the form
described here and compiled on the ground for one sensor profile into a static
flight graph. This document describes the training form, the symmetry it
respects, the capability gating of its heads and its flight budget.

Requirements: MOD-REQ-001 to MOD-REQ-011 and MOD-REQ-060 to MOD-REQ-067 in
[REQUIREMENTS.md](REQUIREMENTS.md). The visual diagrams live in
theoros-brand; the data flow is described in text below.

## Data flow

For a batch of B tiles of N bands, H x W pixels (256 x 256 in flight):

| Step | Module | Output |
|---|---|---|
| 1 | Spectral stem: per-band 1 x 1 kernels generated from band metadata | (B, C_s, H, W) trivial |
| 2 | D4 lift: 4 x 4 kernel at stride 4, batch norm, ReLU6 | (B, 8 F_1, H/4, W/4) regular |
| 3 | Stage 1: inverted residual blocks at stride 4 | (B, 8 F_1, H/4, W/4) |
| 4 | Stage 2: first block downsamples, stride 8 | (B, 8 F_2, H/8, W/8) |
| 5 | Stage 3: first block downsamples, stride 16 | (B, 8 F_3, H/16, W/16) |
| 6 | Feature pyramid: laterals, nearest upsampling, sum, smoothing | (B, 8 P, H/4, W/4) |
| 7 | Four heads | (B, K, H/4, W/4) trivial each |

Trivial maps are ordinary images. Regular maps hold `F` fields of 8 channels
each, one per symmetry of the square; see the next section.

## Symmetry: the group D4

An image taken in orbit has no preferred "up": a field rotated by a quarter
turn or mirrored is still a valid field and its cloud mask is the rotated or
mirrored mask. TheorosNet is built so that this holds exactly (MOD-REQ-063).

D4 has 8 elements `g = (m, r)`, index `4 m + r`: optionally mirror the width
axis, then rotate `r` quarter turns counter-clockwise. The multiplication
table is derived from that action, so it cannot disagree with it.

- A trivial map transforms by moving its pixels: `T_g x`.
- A regular map with fields `f(h)`, channel `8 c + h`, transforms by
  `(g . f)(h) = T_g f(g^-1 h)`: pixels move and the 8 channels of each field
  are permuted.

Equivariant layers, each storing only its unique parameters:

| Layer | Expanded weight | Unique parameters |
|---|---|---|
| Lift, trivial to regular | `W[o, h] = T_h psi[o]` | `F_out C_in k^2` |
| Group convolution, regular to regular | `W[o, h, c, h'] = T_h psi[o, c, h^-1 h']` | `F_out F_in 8 k^2` |
| Depthwise, regular | `W[c, h] = T_h psi[c]` | `F k^2` |
| Invariant projection, regular to trivial | `W[k, c, h] = psi[k, c]` | `K F` plus `K` biases |
| Batch norm, regular | one mean, variance, scale and shift per field | `2 F` |

Every layer computes its forward pass with the expanded weight, so the plain
convolutions the compiler emits are the same function.

Spatial rules that keep the symmetry exact:

- Stride 1 kernels are 3 x 3 with padding 1; zero padding is symmetric.
- Downsampling uses 2 x 2 kernels at stride 2 without padding and the lift
  uses 4 x 4 at stride 4. On an even-sized map such a grid is mapped onto
  itself by every symmetry. A 3 x 3 kernel at stride 2 is not, and a test
  demonstrates the break.
- Upsampling is nearest-neighbour doubling.
- Tile sides are square multiples of 16.

Tolerances (MOD-REQ-063): layers with integer-valued weights and inputs are
equivariant bit for bit in float64; the whole network is equivariant within
1e-9 absolute in float64 and 1e-4 relative to the largest output magnitude in
float32. The remaining difference is floating-point summation order.

## Spectral stem

For band `b` with centre wavelength, FWHM and GSD, the stem computes:

1. Features: sines and cosines at 8 octaves of the centre wavelength scaled
   to [0, 1] over 400 to 2400 nm, and the logarithms of FWHM / 100 nm and
   GSD / 10 m.
2. A per-band embedding `e_b` by a two-layer perceptron.
3. A band-set summary: the mean of the embeddings of the real bands and the
   band count `log(N) / log(13)`.
4. A 1 x 1 kernel `w_b` (C_s values) from `e_b` and the summary, and one
   bias vector from the summary.

The output is `sum_b w_b x_b + bias`. It does not depend on band order or on
padded slots (MOD-REQ-060). A 1 x 1 kernel commutes with every spatial
symmetry, so the stem preserves equivariance; GSD therefore shapes the
spectral mixing rather than a spatial filter. For one sensor every kernel is
a constant, and the compiler replaces the hypernetwork by one plain
convolution of N to C_s channels.

The input is TOA reflectance normalised by the data pipeline; the compiler
folds the sensor's reflectance encoding and the normalisation into the stem.

## Backbone, pyramid and heads

Inverted residual block: pointwise expansion by the tier's expansion factor
with batch norm and ReLU6, depthwise filtering with batch norm and ReLU6, and
a pointwise projection with batch norm. The block adds its input when stride
and width are unchanged. Squeeze-and-excitation is not used: it adds global
pooling and gating outside the flight operator set.

The feature pyramid projects each stage to `P` fields, adds the nearest
upsampled coarser level, and smooths the stride 4 result with a depthwise and
a pointwise convolution.

Each head applies a depthwise and a pointwise convolution on the pyramid and
an invariant projection to its outputs (MOD-REQ-064):

| Head | Channels | Meaning |
|---|---|---|
| `cloud` | 4 | logits for clear, thick cloud, thin cloud, cloud shadow |
| `land_water` | 1 | water logit |
| `vessel` | 3 | vessel centre logit, log length in m, log width in m |
| `fire` | 1 | active fire logit |

Outputs are at stride 4 (64 x 64 for a 256 x 256 tile). Vessel length and
width are read at the peaks of the centre map.

## Capability gating

A head runs only when the sensor has the bands its task physically needs
(MOD-REQ-004, MOD-REQ-065). Each need is met by any band whose centre lies in
one of its windows, inclusive, with a GSD at or below the stated limit.

| Head | Need | Windows | Physical reason |
|---|---|---|---|
| cloud | 1 | blue 420 to 520 nm | cloud and haze scatter strongly in the blue, where clear land and water are dark |
| cloud | 2 | red or NIR 620 to 900 nm | clouds stay bright and spectrally flat from the visible to the near infrared, unlike vegetation, water and soil |
| land_water | 1 | green 520 to 600 nm | water reflects most in the green, the reference band of water indices |
| land_water | 2 | NIR 760 to 900 nm, or SWIR 1550 to 1750 nm | water absorbs near-infrared and shortwave infrared light almost completely, while land reflects it |
| vessel | 1 | NIR 760 to 900 nm with GSD at most 10 m | vessels are brightest against water in the near infrared; a 20 m vessel covers at least two pixels only at 10 m GSD or finer |
| fire | 1 | SWIR 2080 to 2350 nm | thermal emission of flaming fire raises 2.2 um radiance far above reflected sunlight |
| fire | 2 | SWIR 1550 to 1750 nm | 1.6 um rises less with fire and provides the reference that separates fire from bright surfaces |

Known limitation: without a SWIR band, snow and ice can be confused with
cloud. This lowers cloud accuracy but does not disable the head.

The command `theoros sensors show <profile>` prints the assessment with the
band that meets each need, or each missing need with its reason. In training
form the model reports a per-sample capability mask and skips a head no
sample supports; outputs of unsupported heads must be ignored. The compiler
removes disabled heads from the flight graph.

## Tiers and flight budget

| Tier | C_s | Stage fields | Blocks | Expansion | Pyramid | Head |
|---|--:|---|---|--:|--:|--:|
| nano | 12 | 2, 4, 6 | 2, 3, 4 | 2 | 4 | 3 |
| base | 16 | 8, 12, 16 | 2, 3, 4 | 2 | 12 | 3 |
| max | 16 | 12, 16, 24 | 2, 3, 4 | 2 | 12 | 4 |

`budget.py` counts, from the tier definition alone, the unique training
parameters and the flight graph after batch norm folding and D4 expansion,
with int8 weights, int32 biases and int8 activations (MOD-REQ-066,
MOD-REQ-067). A test checks the counts against the built network. Output of
`theoros budget` for 13 bands, all heads and 256 x 256 tiles:

```text
tier   bands  training   flight  nominal  dev %   int8 B     MMAC  peak act B  verdict
nano      13     10396    63193    63000   0.31    68872     84.2     1638400  within budget
base      13     64776   453185   450000   0.71   469484    559.4     1900544  within budget
max       13    119705   890601   890000   0.07   912948    995.1     1966080  within budget
limits: flight parameters below 1000000, int8 weights and int32 biases at most 1048576 B, tier within 5 % of nominal
```

Because every D4 layer expands roughly eightfold, the flight graph has about
six to eight times more stored values than the training form has unique
parameters. The tier sizes and the 1 MiB limit refer to the flight graph.
Peak activation memory assumes layer-by-layer execution with the int8 input
tile resident until the stem has read it. At 13 bands the input tile and the
full-resolution stem output dominate the peak.
