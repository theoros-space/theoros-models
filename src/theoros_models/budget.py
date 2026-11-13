# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Analytic parameter, MAC, memory and int8 size accounting (MOD-REQ-066, MOD-REQ-067).

The budget is computed from a :class:`~theoros_models.arch.TierSpec` alone,
without building a network. It describes two forms:

* the training form, whose parameters are the unique learnable values,
  including the spectral hypernetwork and batch normalisation;
* the flight form after compilation: the hypernetwork replaced by one 1 x 1
  convolution for a fixed band count, batch normalisation folded into
  convolution biases, and every D4 layer expanded to a plain convolution.
  Weights are stored as int8 and biases as int32; activations are int8.

Peak activation memory assumes the flight graph runs layer by layer in the
order listed, holding every tensor from its producer until its last
consumer, with the input tile resident until the stem has consumed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from theoros_models.arch import LIFT_KERNEL, TILE_SIZE, TierSpec
from theoros_models.layers.d4 import ORDER
from theoros_models.layers.heads import HEAD_SPECS

INT8_BYTES = 1
INT32_BYTES = 4
BN_PARAMETERS = 2


@dataclass(frozen=True)
class Op:
    """One flight-graph operation.

    Attributes:
        name: Unique name, for example ``stage2.block0.expand``.
        kind: ``conv``, ``depthwise``, ``add`` or ``upsample``.
        inputs: Names of the tensors read.
        shape: Output shape (C, H, W).
        weights: int8 weight count.
        biases: int32 bias count.
        macs: Multiply-accumulates for one tile.
    """

    name: str
    kind: str
    inputs: tuple[str, ...]
    shape: tuple[int, int, int]
    weights: int = 0
    biases: int = 0
    macs: int = 0

    @property
    def elements(self) -> int:
        """Elements of the output tensor."""
        channels, height, width = self.shape
        return channels * height * width


@dataclass(frozen=True)
class Budget:
    """Budget of one tier for one band count.

    Attributes:
        tier: Tier name.
        bands: Band count of the compiled stem.
        heads: Heads present in the flight graph.
        training_parameters: Unique learnable parameters of the training form.
        ops: Flight-graph operations in execution order.
    """

    tier: str
    bands: int
    heads: tuple[str, ...]
    training_parameters: int
    ops: tuple[Op, ...] = field(repr=False)

    @property
    def flight_weights(self) -> int:
        """int8 weights of the flight graph."""
        return sum(op.weights for op in self.ops)

    @property
    def flight_biases(self) -> int:
        """int32 biases of the flight graph."""
        return sum(op.biases for op in self.ops)

    @property
    def flight_parameters(self) -> int:
        """Weights plus biases of the flight graph (MOD-REQ-001)."""
        return self.flight_weights + self.flight_biases

    @property
    def int8_bytes(self) -> int:
        """Stored size of int8 weights and int32 biases in bytes (MOD-REQ-002)."""
        return INT8_BYTES * self.flight_weights + INT32_BYTES * self.flight_biases

    @property
    def macs(self) -> int:
        """Multiply-accumulates per tile."""
        return sum(op.macs for op in self.ops)

    @property
    def peak_activation_bytes(self) -> int:
        """Largest total size of simultaneously live int8 activations."""
        last_use: dict[str, int] = {}
        for index, op in enumerate(self.ops):
            for name in op.inputs:
                last_use[name] = index
        sizes = {op.name: op.elements * INT8_BYTES for op in self.ops}
        live: set[str] = set()
        peak = 0
        for index, op in enumerate(self.ops):
            live.add(op.name)
            peak = max(peak, sum(sizes[name] for name in live))
            live = {name for name in live if last_use.get(name, -1) > index}
        return peak


class _Graph:
    """Accumulates flight ops and training parameters while walking a tier."""

    def __init__(self) -> None:
        self.ops: list[Op] = []
        self.training = 0

    def add(self, op: Op, training: int = 0) -> str:
        self.ops.append(op)
        self.training += training
        return op.name

    def shape(self, name: str) -> tuple[int, int, int]:
        return next(op.shape for op in self.ops if op.name == name)

    def conv_bn(self, name: str, source: str, out_fields: int, kernel: int = 1) -> str:
        in_channels, height, width = self.shape(source)
        out_channels = ORDER * out_fields
        weights = out_channels * in_channels * kernel * kernel
        training = weights // ORDER + BN_PARAMETERS * out_fields
        op = Op(
            name,
            "conv",
            (source,),
            (out_channels, height, width),
            weights,
            out_channels,
            weights * height * width,
        )
        return self.add(op, training)

    def depthwise_bn(self, name: str, source: str, stride: int = 1) -> str:
        channels, height, width = self.shape(source)
        kernel = 3 if stride == 1 else 2
        out_h, out_w = height // stride, width // stride
        weights = channels * kernel * kernel
        training = weights // ORDER + BN_PARAMETERS * channels // ORDER
        op = Op(
            name,
            "depthwise",
            (source,),
            (channels, out_h, out_w),
            weights,
            channels,
            weights * out_h * out_w,
        )
        return self.add(op, training)


def _hypernetwork_parameters(spec: TierSpec) -> int:
    features = 2 * spec.hyper_frequencies + 2
    embed = features * spec.hyper_embed + spec.hyper_embed
    embed += spec.hyper_embed * spec.hyper_embed + spec.hyper_embed
    kernel = (2 * spec.hyper_embed + 1) * spec.hyper_hidden + spec.hyper_hidden
    kernel += spec.hyper_hidden * spec.stem_channels + spec.stem_channels
    bias = (spec.hyper_embed + 1) * spec.stem_channels + spec.stem_channels
    return embed + kernel + bias


def compute_budget(
    spec: TierSpec,
    bands: int = 13,
    heads: tuple[str, ...] | None = None,
    tile_size: int = TILE_SIZE,
) -> Budget:
    """Compute the budget of a tier.

    Args:
        spec: Tier architecture.
        bands: Band count of the compiled stem, 3 to 13.
        heads: Heads kept in the flight graph; all heads when None.
        tile_size: Input tile edge in pixels, a multiple of 16.

    Returns:
        The budget.
    """
    specs = {head.name: head for head in HEAD_SPECS}
    kept = tuple(specs) if heads is None else heads
    unknown = sorted(set(kept) - set(specs))
    if unknown:
        raise ValueError(f"unknown heads: {', '.join(unknown)}")
    graph = _Graph()
    t = tile_size
    graph.add(Op("input", "input", (), (bands, t, t)))

    stem_weights = bands * spec.stem_channels
    graph.add(
        Op(
            "stem",
            "conv",
            ("input",),
            (spec.stem_channels, t, t),
            stem_weights,
            spec.stem_channels,
            stem_weights * t * t,
        ),
        _hypernetwork_parameters(spec),
    )

    fields = spec.stage_fields[0]
    lift_out = ORDER * fields
    lift_weights = lift_out * spec.stem_channels * LIFT_KERNEL * LIFT_KERNEL
    s = t // LIFT_KERNEL
    current = graph.add(
        Op(
            "lift",
            "conv",
            ("stem",),
            (lift_out, s, s),
            lift_weights,
            lift_out,
            lift_weights * s * s,
        ),
        lift_weights // ORDER + BN_PARAMETERS * fields,
    )

    stage_outputs = []
    in_fields = fields
    for stage, (out_fields, blocks) in enumerate(
        zip(spec.stage_fields, spec.stage_blocks, strict=True)
    ):
        for block in range(blocks):
            prefix = f"stage{stage + 1}.block{block}"
            stride = 2 if stage > 0 and block == 0 else 1
            hidden = in_fields * spec.expansion
            x = graph.conv_bn(f"{prefix}.expand", current, hidden)
            x = graph.depthwise_bn(f"{prefix}.depthwise", x, stride)
            x = graph.conv_bn(f"{prefix}.project", x, out_fields)
            if stride == 1 and in_fields == out_fields:
                x = graph.add(Op(f"{prefix}.add", "add", (current, x), graph.shape(x)))
            current, in_fields = x, out_fields
        stage_outputs.append(current)

    top = graph.conv_bn("neck.lateral3", stage_outputs[2], spec.neck_fields)
    for level in (2, 1):
        channels, height, width = graph.shape(top)
        up = graph.add(Op(f"neck.up{level}", "upsample", (top,), (channels, 2 * height, 2 * width)))
        lateral = graph.conv_bn(f"neck.lateral{level}", stage_outputs[level - 1], spec.neck_fields)
        top = graph.add(Op(f"neck.add{level}", "add", (lateral, up), graph.shape(lateral)))
    neck = graph.depthwise_bn("neck.smooth_depthwise", top)
    neck = graph.conv_bn("neck.smooth_pointwise", neck, spec.neck_fields)

    for name in kept:
        x = graph.depthwise_bn(f"head.{name}.depthwise", neck)
        x = graph.conv_bn(f"head.{name}.pointwise", x, spec.head_fields)
        channels, height, width = graph.shape(x)
        outputs = len(specs[name].channels)
        weights = outputs * channels
        graph.add(
            Op(
                f"head.{name}.output",
                "conv",
                (x,),
                (outputs, height, width),
                weights,
                outputs,
                weights * height * width,
            ),
            weights // ORDER + outputs,
        )
    return Budget(spec.name, bands, kept, graph.training, tuple(graph.ops))
