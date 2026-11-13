# Training

This document describes how TheorosNet is trained. This revision covers the
data pipeline and the losses; the training loop, distributed training and
checkpoints follow with their implementation.

Requirements: MOD-REQ-100 to MOD-REQ-111 and MOD-REQ-120 to MOD-REQ-121 in
[REQUIREMENTS.md](REQUIREMENTS.md).

## Data pipeline

Data flows through five stages:

1. **Dataset adapter.** Returns one `Sample`: TOA reflectance (N, H, W), the
   N bands, the sensor name and optional labels. Adapters exist for the tile
   store and the synthetic generator; each names a card in `datasets/`.
2. **Augmentation.** A random symmetry of the square applied to image, dense
   labels and vessel centres; per-band gain (standard deviation 3 percent)
   and offset (0.005 reflectance) with pixel noise (0.003); band dropout
   with probability 0.25, keeping 3 or more bands in their original order.
3. **Sampling.** `BalancedSampler` groups samples by sensor and labelled
   tasks, draws a group uniformly or by weight, then a sample within it. The
   sequence depends only on seed and epoch, and ranks take interleaved
   shares of it.
4. **Packing.** `collate` normalises reflectance as
   `(clip(reflectance, 0, 1.5) - 0.2) / 0.2`, pads band channels to the
   largest count with a band mask, fills missing dense labels with 255 and
   pads vessel lists.
5. **Loss.** `multitask_loss` evaluates each head on the samples that carry
   its label and whose sensor supports it.

### Labels

| Task | Label | Head output |
|---|---|---|
| cloud | uint8 (H, W): 0 clear, 1 thick, 2 thin, 3 shadow | 4 logits at stride 4 |
| land and water | uint8 (H, W): 0 land, 1 water | 1 logit at stride 4 |
| vessel | float32 (K, 4): centre row and column in pixels, length and width in m | centre logit, log length, log width |
| fire | uint8 (H, W): 0 none, 1 active fire | 1 logit at stride 4 |

The value 255 marks pixels without a valid label. A task with no label at
all is unlabelled for the sample.

### Losses

Dense logits are upsampled bilinearly by 4 to label resolution, which
commutes with the symmetries of the square.

| Head | Loss | Defaults |
|---|---|---|
| cloud | cross-entropy, ignore 255, optional class weights | unweighted |
| land and water | binary cross-entropy, ignore 255 | positive weight 1 |
| fire | binary cross-entropy, ignore 255 | positive weight 10 |
| vessel | penalty-reduced focal loss (alpha 2, beta 4) on a Gaussian centre heatmap with sigma 1 output pixel, plus 0.1 times L1 on log length and width at centres | |

Each per-sample loss is averaged over valid pixels or vessel centres. A
head's loss is the mean over its active samples, and the total is the
weighted sum over heads. Inactive samples contribute zero loss and zero
gradient. The vessel and fire output biases start at the logit of 0.01 so
that training begins near the background rate.

### Synthetic data

`SyntheticDataset` renders scenes from analytic material spectra for any
sensor profile, so tests and CI never download data. Its card,
[datasets/synthetic.md](../datasets/synthetic.md), records how it is built
and why its results say nothing about real-world accuracy.

### Real data

External datasets are converted into the tile store described in
[datasets/README.md](../datasets/README.md) once their card, with a verified
license, exists. The test split of a store opens only with
`evaluation=True`.
