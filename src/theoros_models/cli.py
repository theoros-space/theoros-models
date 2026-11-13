# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""The ``theoros`` command line (MOD-REQ-220)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from theoros_models.arch import (
    FLIGHT_INT8_BYTES_LIMIT,
    FLIGHT_PARAMETER_LIMIT,
    TIER_TOLERANCE,
    TIERS,
)
from theoros_models.budget import compute_budget
from theoros_models.capabilities import assess_profile
from theoros_models.sensors import (
    ProfileError,
    SensorProfile,
    catalogue_names,
    load_catalogue_profile,
    load_profile,
    resolve_profile,
)
from theoros_models.version import __version__

EXIT_OK = 0
EXIT_INVALID = 1


def _format_profile(profile: SensorProfile) -> str:
    offset = profile.encoding.offset
    sign = "-" if offset < 0 else "+"
    lines = [
        f"name         {profile.name}",
        f"platform     {profile.platform}",
        f"instrument   {profile.instrument}",
        f"fingerprint  {profile.fingerprint()}",
        f"encoding     reflectance = {profile.encoding.scale:g} * value {sign} {abs(offset):g}",
        "",
        f"{'band':<14}{'centre nm':>10}{'FWHM nm':>10}{'GSD m':>8}  srf",
    ]
    for band in profile.bands:
        srf = "tabulated" if band.srf is not None else "gaussian"
        lines.append(
            f"{band.name:<14}{band.centre_nm:>10.1f}{band.fwhm_nm:>10.1f}{band.gsd_m:>8.1f}  {srf}"
        )
    lines += ["", "capabilities", assess_profile(profile).format()]
    return "\n".join(lines)


def _budget(args: argparse.Namespace) -> int:
    names = list(TIERS) if args.tier == "all" else [args.tier]
    print(
        f"{'tier':<6}{'bands':>6}{'training':>10}{'flight':>9}{'nominal':>9}{'dev %':>7}"
        f"{'int8 B':>9}{'MMAC':>9}{'peak act B':>12}  verdict"
    )
    status = EXIT_OK
    for name in names:
        spec = TIERS[name]
        budget = compute_budget(spec, bands=args.bands)
        deviation = 100.0 * (budget.flight_parameters / spec.nominal_parameters - 1.0)
        ok = (
            budget.flight_parameters < FLIGHT_PARAMETER_LIMIT
            and budget.int8_bytes <= FLIGHT_INT8_BYTES_LIMIT
            and abs(deviation) <= 100.0 * TIER_TOLERANCE
        )
        status = status if ok else EXIT_INVALID
        print(
            f"{name:<6}{args.bands:>6}{budget.training_parameters:>10}"
            f"{budget.flight_parameters:>9}{spec.nominal_parameters:>9}{deviation:>7.2f}"
            f"{budget.int8_bytes:>9}{budget.macs / 1e6:>9.1f}{budget.peak_activation_bytes:>12}"
            f"  {'within budget' if ok else 'OVER BUDGET'}"
        )
    print(
        f"limits: flight parameters below {FLIGHT_PARAMETER_LIMIT}, int8 weights and int32 "
        f"biases at most {FLIGHT_INT8_BYTES_LIMIT} B, tier within "
        f"{100.0 * TIER_TOLERANCE:g} % of nominal"
    )
    return status


def _sensors_list(_: argparse.Namespace) -> int:
    print(f"{'name':<24}{'bands':>6}  instrument")
    for name in catalogue_names():
        profile = load_catalogue_profile(name)
        print(f"{name:<24}{len(profile.bands):>6}  {profile.platform} {profile.instrument}")
    return EXIT_OK


def _sensors_show(args: argparse.Namespace) -> int:
    try:
        profile = resolve_profile(args.profile)
    except ProfileError as error:
        _report(error)
        return EXIT_INVALID
    print(_format_profile(profile))
    return EXIT_OK


def _sensors_validate(args: argparse.Namespace) -> int:
    status = EXIT_OK
    for path in args.paths:
        try:
            profile = load_profile(path)
        except ProfileError as error:
            _report(error)
            status = EXIT_INVALID
            continue
        print(f"{path}: valid, {len(profile.bands)} bands, {profile.fingerprint()}")
    return status


def _report(error: ProfileError) -> None:
    print(f"{error.source}: invalid", file=sys.stderr)
    for problem in error.problems:
        print(f"  {problem}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="theoros", description="TheorosNet training, compilation and export tools."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    sensors = commands.add_parser("sensors", help="inspect and validate sensor profiles")
    actions = sensors.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="list the packaged catalogue").set_defaults(
        handler=_sensors_list
    )
    show = actions.add_parser("show", help="show one profile and its fingerprint")
    show.add_argument("profile", help="catalogue name or path to a profile YAML file")
    show.set_defaults(handler=_sensors_show)
    validate = actions.add_parser("validate", help="validate profile YAML files")
    validate.add_argument("paths", nargs="+", help="profile YAML files")
    validate.set_defaults(handler=_sensors_validate)

    budget = commands.add_parser("budget", help="print the analytic flight budget of the tiers")
    budget.add_argument("--tier", choices=[*TIERS, "all"], default="all")
    budget.add_argument(
        "--bands",
        type=int,
        choices=range(3, 14),
        default=13,
        metavar="{3..13}",
        help="band count of the compiled stem (default 13, the largest)",
    )
    budget.set_defaults(handler=_budget)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit status.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 when an input is invalid, 2 on a usage error.
    """
    args = build_parser().parse_args(argv)
    status: int = args.handler(args)
    return status
