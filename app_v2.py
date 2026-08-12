"""HydroStar NextGen Electrolyser Testing and Validation — local Demo mode.

This dashboard simulates the 16 process instruments shown on NX-PID-001 so the
layout, alarm colours, alarm thresholds, trend charts, sensor table and event
behaviour can be reviewed before the physical sensors and the final live-data
architecture exist. It runs entirely offline:

    streamlit run app_v2.py

No Supabase, Raspberry Pi, secrets, credentials or network access is used or
required. Data provenance remains an internal implementation detail and is not
shown in the operator interface.

Design notes for the next developer
-----------------------------------
* Internal sensor keys (for example ``FT003_O2`` and ``FT003_N2``) are used as
dataframe keys instead of the displayed P&ID tags because NX-PID-001 shows
``FT-003`` on both the oxygen product stream and the nitrogen feed. Keying on
the displayed tag would silently overwrite one instrument with the other.
* The P&ID labels the two gas analysers without unique numeric instrument
tags. Working dashboard tags ``AIT-001`` and ``AIT-002`` distinguish them
until those identifiers are confirmed on a future drawing revision.
* Instruments whose four limits are all null are shown grey as
``No approved limits`` (UNCONFIGURED), never green NORMAL: a green state
would wrongly imply the value had been checked against approved limits.
* Demo history is stable across Streamlit reruns because every sample is a
pure function of (scenario seed, sensor key, sample index). Widget-driven
reruns therefore recompute identical values; only genuinely new five-second
samples are appended (see ``advance_sim``).
* The alarm-event log cannot duplicate events on refresh because samples are
processed exactly once (a ``last_k`` pointer) and an event is recorded only
when a sensor's state differs from its previously recorded state
(see ``collect_events``).

P&ID context help uses one master image at ``assets/pid/NX-PID-001.jpg``.
Each sensor's responsive highlight coordinates live in the YAML registry, so
clicking a locator control opens the same drawing with only that sensor highlighted.

All alarm limits in this file and in the registry are provisional dashboard
values for layout and behaviour testing. They are not PLC, SIS or approved
plant trip settings, and this dashboard issues no control commands.
"""

from __future__ import annotations

import base64
import html
import math
import re
import zlib
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from string import Template
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

try:
    from zoneinfo import ZoneInfo

    DISPLAY_TZ = ZoneInfo("Europe/London")
    DISPLAY_TZ_NAME = "Europe/London"
except Exception:  # pragma: no cover - zoneinfo ships with Python 3.11+
    DISPLAY_TZ = timezone.utc
    DISPLAY_TZ_NAME = "UTC"


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
LOGO_FILE = APP_DIR / "logo.png"
REGISTRY_FILE = APP_DIR / "provisional_sensor_registry_v2.yaml"
PID_IMAGE_FILE = APP_DIR / "assets" / "pid" / "NX-PID-001.jpg"

# The P&ID highlight coordinates are percentages of the full master image.
# Circular instrument bubbles use this default diameter; rectangular items can
# override width/height in provisional_sensor_registry_v2.yaml.
DEFAULT_PID_CIRCLE_WIDTH = 3.2

PAGE_TITLE = "NextGen Electrolyser Testing and Validation"
PAGE_SUBTITLE = "5 kW Electrolyser Testing"

SAMPLE_SECONDS = 5
REFRESH_INTERVAL = "5s"
HISTORY_SECONDS = 6 * 3600
HISTORY_MINUTES = HISTORY_SECONDS / 60.0
HISTORY_SAMPLES = HISTORY_SECONDS // SAMPLE_SECONDS  # 4,320 samples
EVENT_LIMIT = 100
MAX_CHART_POINTS = 1_200

ACTIVE_MAX_AGE_SECONDS = 20.0
DELAYED_MAX_AGE_SECONDS = 60.0

# HydroStar brand palette (Section 16 of the build brief).
PRIMARY = "#A7D730"
SECONDARY = "#499823"
DARK_GREY = "#30343C"
LIGHT_GREY = "#8C919A"
WATER_BLUE = "#4EA8DE"
AMBER = "#F6A609"
RED = "#D32F2F"
BADGE_DARK_TEXT = "#1A2010"

FONT_STACK = "'Hind', 'Inter', Arial, sans-serif"

SOURCE_LOCAL = "LOCAL DATA"
QUALITY_GOOD = "GOOD"
QUALITY_STALE = "STALE"
QUALITY_MISSING = "MISSING"
QUALITY_BAD = "BAD"
BAD_QUALITY = frozenset({QUALITY_STALE, QUALITY_MISSING, QUALITY_BAD})

STATE_NORMAL = "NORMAL"
STATE_UNKNOWN = "UNKNOWN"
STATE_UNCONFIGURED = "UNCONFIGURED"

# Deterministic worst-first tie ordering for safety summaries (Section 8).
ALARM_PRECEDENCE: tuple[str, ...] = (
    "HH", "LL", "H", "L", STATE_NORMAL, STATE_UNCONFIGURED, STATE_UNKNOWN,
)
_PRECEDENCE_RANK = {state: rank for rank, state in enumerate(ALARM_PRECEDENCE)}

CRITICAL_STATES = frozenset({"HH", "LL"})
WARNING_STATES = frozenset({"H", "L"})

STATE_COLOURS = {
    STATE_NORMAL: SECONDARY,
    "L": AMBER,
    "H": AMBER,
    "LL": RED,
    "HH": RED,
    STATE_UNKNOWN: LIGHT_GREY,
    STATE_UNCONFIGURED: LIGHT_GREY,
}
SEVERITY_WORD = {
    "HH": "Critical",
    "LL": "Critical",
    "H": "Warning",
    "L": "Warning",
    STATE_NORMAL: "Normal",
    STATE_UNCONFIGURED: "Limits pending",
    STATE_UNKNOWN: "Unavailable",
}
ALARM_LABEL = {
    "HH": "High high",
    "LL": "Low low",
    "H": "High",
    "L": "Low",
    STATE_NORMAL: "Normal",
    STATE_UNCONFIGURED: "Limits pending",
    STATE_UNKNOWN: "Unavailable",
}

NO_APPROVED_LIMITS = "No approved limits"

MEASUREMENT_DECIMALS = {
    "level_deviation": 1,
    "temperature": 1,
    "pressure": 2,
    "flow": 2,
    "gas_concentration": 2,
    "conductivity": 0,
}
MEASUREMENT_LINE_COLOURS = {
    "level_deviation": SECONDARY,
    "temperature": PRIMARY,
    "pressure": WATER_BLUE,
    "flow": WATER_BLUE,
    "gas_concentration": SECONDARY,
    "conductivity": WATER_BLUE,
}

# Internal keys in registry order. Displayed P&ID tags are NOT used as keys
# because FT-003 appears twice on NX-PID-001 (oxygen product outlet and
# nitrogen feed) and would overwrite itself in any tag-keyed structure.
EXPECTED_KEYS: tuple[str, ...] = (
    "LT001", "TT001", "PT001", "LT002",
    "CT001",
    "FT001_H2", "FT002_O2",
    "ANALYSER_H2_IN_O2", "ANALYSER_O2_IN_H2",
    "TT002_O2", "PT002_O2", "FT003_O2", "TT003_H2", "PT003_H2", "FT004_H2",
    "FT003_N2",
)
CRITICAL_CARD_ORDER: tuple[str, ...] = (
    "LT001", "TT001", "PT001", "LT002",
    "FT001_H2", "FT002_O2", "ANALYSER_H2_IN_O2", "ANALYSER_O2_IN_H2",
)

WINDOW_OPTIONS: dict[str, float] = {
    "5 minutes": 5.0,
    "15 minutes": 15.0,
    "1 hour": 60.0,
    "6 hours": 360.0,
}
DEFAULT_WINDOW = "15 minutes"
DEFAULT_TREND_SENSORS = ["LT001", "ANALYSER_H2_IN_O2"]

SAFETY_ITEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Gas crossover", ("ANALYSER_H2_IN_O2", "ANALYSER_O2_IN_H2")),
    ("Pressure limits", ("PT001", "PT002_O2", "PT003_H2")),
    ("Temperature limits", ("TT001", "TT002_O2", "TT003_H2")),
    ("Level limits", ("LT001", "LT002")),
)

CHART_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}


# ---------------------------------------------------------------------------
# 2. Sensor registry (single source of sensor metadata)
# ---------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Raised when the sensor registry is missing, unreadable or invalid."""


@dataclass(frozen=True)
class PidHighlight:
    """Responsive P&ID marker stored as percentages of the master image."""

    x: float
    y: float
    shape: str = "circle"
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass(frozen=True)
class SensorSpec:
    """One process instrument from provisional_sensor_registry_v2.yaml."""

    key: str
    pid_tag: str
    display_name: str
    description: str
    group_id: int
    group_name: str
    measurement: str
    unit: str
    unit_provisional: bool
    critical: bool
    limits: Mapping[str, Optional[float]]
    source_mode: str
    pid_highlight: PidHighlight
    notes: str = ""

    @property
    def display_tag(self) -> str:
        return clean_ui_tag(self.pid_tag)

    @property
    def decimals(self) -> int:
        return MEASUREMENT_DECIMALS.get(self.measurement, 2)

    @property
    def configured(self) -> bool:
        return any(value is not None for value in self.limits.values())


REQUIRED_SENSOR_FIELDS = (
    "internal_key", "pid_tag", "display_name", "description", "group_id",
    "group_name", "measurement", "unit", "critical", "limits", "source_mode",
    "pid_highlight",
)
LIMIT_KEYS = ("LL", "L", "H", "HH")


def load_registry(path: Path) -> list[SensorSpec]:
    """Load and validate the sensor registry, failing with readable messages."""
    if not path.exists():
        raise RegistryError(
            f"The sensor registry file is missing: {path.name}. "
            "Place provisional_sensor_registry_v2.yaml in the same folder as "
            "app_v2.py and reload the page."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(
            f"The sensor registry could not be parsed as YAML: {exc}"
        ) from exc

    sensors_raw = raw.get("sensors") if isinstance(raw, dict) else None
    if not isinstance(sensors_raw, list):
        raise RegistryError(
            "The sensor registry does not contain a 'sensors' list."
        )

    problems: list[str] = []
    specs: list[SensorSpec] = []
    seen_keys: set[str] = set()

    for index, entry in enumerate(sensors_raw, start=1):
        if not isinstance(entry, dict):
            problems.append(f"Sensor entry {index} is not a mapping.")
            continue
        missing = [field for field in REQUIRED_SENSOR_FIELDS if field not in entry]
        if missing:
            label = entry.get("internal_key", f"entry {index}")
            problems.append(
                f"Sensor '{label}' is missing required field(s): "
                + ", ".join(missing)
            )
            continue

        key = str(entry["internal_key"])
        if key in seen_keys:
            problems.append(f"Internal key '{key}' is duplicated in the registry.")
            continue
        seen_keys.add(key)

        limits_raw = entry.get("limits")
        limits: dict[str, Optional[float]] = {}
        if not isinstance(limits_raw, dict):
            problems.append(f"Sensor '{key}' has an invalid 'limits' mapping.")
            continue
        limit_error = False
        for limit_name in LIMIT_KEYS:
            value = limits_raw.get(limit_name)
            if value is None:
                limits[limit_name] = None
            else:
                try:
                    limits[limit_name] = float(value)
                except (TypeError, ValueError):
                    problems.append(
                        f"Sensor '{key}' limit {limit_name} is not numeric."
                    )
                    limit_error = True
        if limit_error:
            continue

        highlight_raw = entry.get("pid_highlight")
        if not isinstance(highlight_raw, dict):
            problems.append(f"Sensor '{key}' has an invalid 'pid_highlight' mapping.")
            continue
        try:
            pid_x = float(highlight_raw["x"])
            pid_y = float(highlight_raw["y"])
        except (KeyError, TypeError, ValueError):
            problems.append(
                f"Sensor '{key}' requires numeric pid_highlight x and y values."
            )
            continue
        if not (0.0 <= pid_x <= 100.0 and 0.0 <= pid_y <= 100.0):
            problems.append(
                f"Sensor '{key}' pid_highlight coordinates must be between 0 and 100."
            )
            continue

        pid_shape = str(highlight_raw.get("shape", "circle")).strip().lower()
        if pid_shape not in {"circle", "rectangle"}:
            problems.append(
                f"Sensor '{key}' pid_highlight shape must be 'circle' or 'rectangle'."
            )
            continue

        def _optional_positive(name: str) -> Optional[float]:
            raw_value = highlight_raw.get(name)
            if raw_value is None:
                return None
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                problems.append(
                    f"Sensor '{key}' pid_highlight {name} must be numeric."
                )
                return None
            if numeric <= 0.0 or numeric > 100.0:
                problems.append(
                    f"Sensor '{key}' pid_highlight {name} must be greater than 0 and at most 100."
                )
                return None
            return numeric

        pid_width = _optional_positive("width")
        pid_height = _optional_positive("height")
        if pid_shape == "rectangle" and (pid_width is None or pid_height is None):
            problems.append(
                f"Sensor '{key}' rectangle highlight requires width and height."
            )
            continue

        specs.append(
            SensorSpec(
                key=key,
                pid_tag=str(entry["pid_tag"]),
                display_name=str(entry["display_name"]),
                description=str(entry["description"]),
                group_id=int(entry["group_id"]),
                group_name=str(entry["group_name"]),
                measurement=str(entry["measurement"]),
                unit=str(entry["unit"]),
                unit_provisional=str(entry.get("unit_status", "")) == "provisional",
                critical=bool(entry["critical"]),
                limits=limits,
                source_mode=str(entry["source_mode"]),
                pid_highlight=PidHighlight(
                    x=pid_x,
                    y=pid_y,
                    shape=pid_shape,
                    width=pid_width,
                    height=pid_height,
                ),
                notes=str(entry.get("notes") or ""),
            )
        )

    if len(sensors_raw) != 16:
        problems.append(
            f"The registry must contain exactly 16 sensors, found {len(sensors_raw)}."
        )
    missing_keys = [key for key in EXPECTED_KEYS if key not in seen_keys]
    unexpected_keys = sorted(seen_keys - set(EXPECTED_KEYS))
    if missing_keys:
        problems.append("Missing expected internal key(s): " + ", ".join(missing_keys))
    if unexpected_keys:
        problems.append("Unexpected internal key(s): " + ", ".join(unexpected_keys))

    if problems:
        raise RegistryError(
            "The sensor registry failed validation:\n- " + "\n- ".join(problems)
        )

    # Stable presentation order: group_id first, then registry order.
    specs.sort(key=lambda spec: spec.group_id)
    return specs


# ---------------------------------------------------------------------------
# 3. Alarm engine (the only place threshold comparisons live)
# ---------------------------------------------------------------------------


def classify_alarm(
    value: Optional[float],
    limits: Mapping[str, Optional[float]],
    quality: str,
) -> str:
    """Classify one reading into an alarm state.

    Rules (in order):
    1. MISSING / STALE / BAD quality  -> UNKNOWN (quality takes priority).
    2. Missing or non-numeric value   -> UNKNOWN.
    3. All four limits null           -> UNCONFIGURED (shown grey as
        "No approved limits", never green NORMAL).
    4. value <= LL -> LL;  value <= L -> L;
        value >= HH -> HH;  value >= H -> H;  otherwise NORMAL.
        Exact boundary values therefore count as alarm states.

    No hysteresis is applied because no approved hysteresis values exist yet.
    When they do, add optional ``hysteresis`` / ``previous_state`` parameters
    here rather than scattering comparisons through the UI.
    """
    if quality in BAD_QUALITY:
        return STATE_UNKNOWN
    if value is None:
        return STATE_UNKNOWN
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return STATE_UNKNOWN
    if math.isnan(numeric):
        return STATE_UNKNOWN

    low_low = limits.get("LL")
    low = limits.get("L")
    high = limits.get("H")
    high_high = limits.get("HH")
    if low_low is None and low is None and high is None and high_high is None:
        return STATE_UNCONFIGURED

    if low_low is not None and numeric <= low_low:
        return "LL"
    if low is not None and numeric <= low:
        return "L"
    if high_high is not None and numeric >= high_high:
        return "HH"
    if high is not None and numeric >= high:
        return "H"
    return STATE_NORMAL


def worst_state(states: Iterable[str]) -> str:
    """Most severe state using the deterministic HH, LL, H, L, ... ordering."""
    ranked = [state for state in states if state in _PRECEDENCE_RANK]
    if not ranked:
        return STATE_UNKNOWN
    return min(ranked, key=_PRECEDENCE_RANK.__getitem__)


def state_badge_colours(state: str) -> tuple[str, str]:
    """Background and text colour for an alarm badge (text always shown too)."""
    background = STATE_COLOURS.get(state, LIGHT_GREY)
    # Green, amber and grey need dark text to meet small-text contrast; the
    # darker critical red remains most legible with white text.
    foreground = (
        BADGE_DARK_TEXT
        if background in {SECONDARY, AMBER, LIGHT_GREY}
        else "#ffffff"
    )
    return background, foreground


def alarm_display_text(state: str) -> str:
    """Return concise, plain-language alarm wording for the interface."""
    return ALARM_LABEL.get(state, state.title())


def level_deviation_from_distance(
    raw_distance_cm: float, baseline_distance_mm: float
) -> float:
    """Future Hybrid-mode conversion for the ultrasonic level sensors.

    ``level_deviation_mm = baseline_distance_mm - (raw_distance_cm * 10)``

    A smaller measured distance means a higher liquid level, so positive
    deviation = level has risen. Demo mode deliberately does NOT call this:
    demo level values are generated directly as deviation in millimetres, and
    no approved baseline distance exists yet for LT-001 or LT-002.
    """
    return baseline_distance_mm - (raw_distance_cm * 10.0)


# ---------------------------------------------------------------------------
# 4. Deterministic simulation engine
# ---------------------------------------------------------------------------
# Every sample is a pure function of (scenario seed, sensor key, sample
# index k), built from:
#   * piecewise-linear status waypoints (smooth trends and limit events),
#   * two slow sensor-specific sinusoids (gentle wobble),
#   * tiny per-sample jitter from a counter-based hash (splitmix64).
# Because nothing depends on wall-clock randomness, Streamlit reruns caused
# by widgets recompute identical history, and appending sample k tomorrow
# gives the same value as appending it today.

_SM_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_M2 = np.uint64(0x94D049BB133111EB)
_SALT_PRIME = np.uint64(0xD1B54A32D192ED03)


def _mix64(values: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser over uint64 values."""
    with np.errstate(over="ignore"):
        mixed = values + _SM_GAMMA
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _SM_M1
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _SM_M2
        return mixed ^ (mixed >> np.uint64(31))


def _stream_seed(scenario_seed: int, key: str) -> np.uint64:
    """Stable 64-bit stream seed for one (scenario, sensor) pair."""
    with np.errstate(over="ignore"):
        combined = (
            np.uint64(scenario_seed) * np.uint64(0x100000001B3)
        ) ^ np.uint64(zlib.crc32(key.encode("utf-8")))
    return _mix64(np.asarray([combined], dtype=np.uint64))[0]


def _uniform(seed: np.uint64, ks: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic uniforms in [0, 1) for the given sample indices."""
    with np.errstate(over="ignore"):
        counters = ks.astype(np.uint64) * _SM_GAMMA
        counters = counters ^ seed ^ (np.uint64(salt) * _SALT_PRIME)
    hashed = _mix64(counters)
    return (hashed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _gaussian(seed: np.uint64, ks: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic standard-normal jitter (Box-Muller, clipped)."""
    u1 = np.clip(_uniform(seed, ks, salt), 1e-12, 1.0)
    u2 = _uniform(seed, ks, salt ^ 0x5851F42D)
    normal = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return np.clip(normal, -3.5, 3.5)


def _wobble(seed: np.uint64, t_min: np.ndarray, amplitude: float) -> np.ndarray:
    """Two slow sensor-specific sinusoids: smooth, deterministic variation."""
    if amplitude <= 0:
        return np.zeros_like(t_min)
    params = _uniform(seed, np.arange(4, dtype=np.int64), 0x0BADCAFE)
    period_1 = 18.0 + 24.0 * params[0]   # 18-42 minutes
    period_2 = 6.0 + 7.0 * params[1]     # 6-13 minutes
    phase_1 = 2.0 * np.pi * params[2]
    phase_2 = 2.0 * np.pi * params[3]
    return amplitude * (
        0.7 * np.sin(2.0 * np.pi * t_min / period_1 + phase_1)
        + 0.3 * np.sin(2.0 * np.pi * t_min / period_2 + phase_2)
    )


@dataclass(frozen=True)
class QualityRule:
    """Marks a sensor's samples MISSING or STALE inside a time window.

    ``start_min`` / ``end_min`` are minutes relative to the moment the demo
    session was generated (negative = in the pre-generated history,
    ``end_min=None`` = open-ended into live time).
    """

    sensor: str
    kind: str
    start_min: float
    end_min: Optional[float] = None


@dataclass(frozen=True)
class Scenario:
    """One demo scenario: seed, waypoint profiles and data-quality rules."""

    name: str
    seed: int
    summary: str
    waypoints: Mapping[str, tuple[tuple[float, float], ...]]
    quality_rules: tuple[QualityRule, ...] = ()


# Quiet baselines used whenever a scenario does not override a sensor.
BASELINES: dict[str, float] = {
    "LT001": 0.4,
    "LT002": -0.6,
    "TT001": 44.5,
    "PT001": 1.80,
    "CT001": 500.0,
    "FT002_O2": 0.82,
    "ANALYSER_H2_IN_O2": 0.90,
    "ANALYSER_O2_IN_H2": 1.15,
    "TT002_O2": 43.0,
    "PT002_O2": 1.95,
    "TT003_H2": 45.0,
    "PT003_H2": 1.90,
    "FT003_N2": 4.0,
}

# Hydrogen-side and outlet flows are derived from the oxygen product flow so
# the streams stay physically related: FT-001 tracks roughly twice FT-002 with
# a small dynamic lag plus its own noise (never mathematically identical).
_DERIVED_FLOWS: dict[str, tuple[str, float, float]] = {
    # key: (source key, multiplier, lag seconds)
    "FT001_H2": ("FT002_O2", 2.00, 20.0),
    "FT004_H2": ("FT002_O2", 1.93, 45.0),
    "FT003_O2": ("FT002_O2", 0.95, 30.0),
}

# (wobble amplitude, per-sample jitter sigma) in engineering units.
_WOBBLE_NOISE_BY_MEASUREMENT: dict[str, tuple[float, float]] = {
    "level_deviation": (0.55, 0.04),
    "temperature": (0.85, 0.06),
    "pressure": (0.05, 0.006),
    "gas_concentration": (0.07, 0.010),
    "flow": (0.035, 0.006),
    "conductivity": (7.0, 0.8),
}
_WOBBLE_NOISE_OVERRIDES: dict[str, tuple[float, float]] = {
    "FT001_H2": (0.06, 0.010),   # twice the O2 magnitude, so slightly larger
    "FT004_H2": (0.05, 0.009),
    "FT003_N2": (0.22, 0.03),    # SLPM scale differs from Nm3/h flows
}

# Demo-only electrical channels (not part of the 16 P&ID process sensors).
_ELECTRICAL_BASE: dict[str, tuple[float, float, float]] = {
    # pseudo key: (baseline, wobble amplitude, jitter sigma)
    "STACK_VOLTAGE": (65.6, 0.45, 0.05),
    "STACK_CURRENT": (74.5, 1.10, 0.15),
}


def _scenario(name: str, seed: int, summary: str,
            waypoints: Mapping[str, tuple[tuple[float, float], ...]],
            quality_rules: tuple[QualityRule, ...] = ()) -> Scenario:
    return Scenario(name=name, seed=seed, summary=summary,
                    waypoints=dict(waypoints), quality_rules=quality_rules)


SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        _scenario(
            "Operating normally", 101,
            "All configured sensors stay below their provisional limits.",
            {},
        ),
        _scenario(
            "Multiple alarms active", 202,
            "Green, amber, red and grey states are visible together.",
            {
                "LT001": ((-360.0, 0.4), (-235.0, 0.4), (-208.0, 7.0)),
                "LT002": ((-360.0, -0.6), (-218.0, -0.6), (-188.0, -11.3)),
                "TT001": ((-360.0, 46.0),),
                "PT001": ((-360.0, 1.80), (-168.0, 1.80), (-142.0, 2.82)),
                "FT002_O2": ((-360.0, 0.82), (-150.0, 0.82), (-120.0, 0.90)),
                "ANALYSER_H2_IN_O2": ((-360.0, 0.90), (-112.0, 0.90), (-72.0, 3.26)),
                "ANALYSER_O2_IN_H2": ((-360.0, 1.20),),
                "TT002_O2": ((-360.0, 43.0), (-152.0, 43.0), (-116.0, 61.8)),
                "PT002_O2": ((-360.0, 2.00),),
                "TT003_H2": ((-360.0, 45.0), (-102.0, 45.0), (-80.0, 54.0)),
                "PT003_H2": ((-360.0, 1.90), (-132.0, 1.90), (-96.0, 4.26)),
            },
        ),
        _scenario(
            "Level limit exceeded", 303,
            "LT001 rises through the high and high high limits, then settles "
            "above the high limit.",
            {
                "LT001": (
                    (-360.0, 0.4), (-95.0, 0.4), (-72.0, 7.2), (-52.0, 11.6),
                    (-34.0, 11.6), (-14.0, 7.2), (20.0, 7.2),
                ),
            },
        ),
        _scenario(
            "Temperature limit exceeded", 404,
            "TT001 ramps NORMAL, H, HH and later moves towards recovery.",
            {
                "TT001": (
                    (-360.0, 44.5), (-80.0, 44.5), (-58.0, 53.5),
                    (-32.0, 61.6), (-6.0, 61.6), (32.0, 46.5),
                ),
            },
        ),
        _scenario(
            "Pressure limit exceeded", 505,
            "PT001 ramps NORMAL, H, HH and later moves towards recovery.",
            {
                "PT001": (
                    (-360.0, 1.80), (-75.0, 1.80), (-52.0, 2.90),
                    (-26.0, 4.30), (-2.0, 4.30), (36.0, 1.90),
                ),
            },
        ),
        _scenario(
            "Gas crossover detected", 606,
            "H2 in O2 rises through H to HH while O2 in H2 stays normal.",
            {
                "ANALYSER_H2_IN_O2": (
                    (-360.0, 0.90), (-85.0, 0.90), (-58.0, 2.35),
                    (-30.0, 3.42), (-8.0, 3.42), (22.0, 2.35), (50.0, 1.20),
                ),
            },
        ),
        _scenario(
            "Sensor data fault active", 707,
            "FT001 loses data gaps and TT002 goes stale; alarms turn grey.",
            {},
            quality_rules=(
                QualityRule("FT001_H2", QUALITY_MISSING, -120.0, -100.0),
                QualityRule("FT001_H2", QUALITY_MISSING, -15.0, None),
                QualityRule("TT002_O2", QUALITY_STALE, -20.0, None),
                QualityRule("LT001", QUALITY_MISSING, -190.0, -178.0),
            ),
        ),
    )
}
STATUS_ORDER: tuple[str, ...] = tuple(SCENARIOS)
DEFAULT_STATUS = "Multiple alarms active"


def _t_minutes(ks: np.ndarray) -> np.ndarray:
    """Sample indices -> minutes relative to the generation moment."""
    return ks.astype(np.float64) * SAMPLE_SECONDS / 60.0 - HISTORY_MINUTES


def _profile(scenario: Scenario, key: str, t_min: np.ndarray) -> np.ndarray:
    """Piecewise-linear waypoint profile (clamped and held at both ends)."""
    waypoints = scenario.waypoints.get(key)
    if not waypoints:
        return np.full(t_min.shape, BASELINES[key], dtype=np.float64)
    xs = np.array([point[0] for point in waypoints], dtype=np.float64)
    ys = np.array([point[1] for point in waypoints], dtype=np.float64)
    return np.interp(t_min, xs, ys)


def _wobble_noise_amplitudes(spec: SensorSpec) -> tuple[float, float]:
    if spec.key in _WOBBLE_NOISE_OVERRIDES:
        return _WOBBLE_NOISE_OVERRIDES[spec.key]
    return _WOBBLE_NOISE_BY_MEASUREMENT.get(spec.measurement, (0.05, 0.01))


def _raw_values(scenario: Scenario, spec: SensorSpec, ks: np.ndarray) -> np.ndarray:
    """Profile + wobble + jitter, before any data-quality effects."""
    t_min = _t_minutes(ks)
    if spec.key in _DERIVED_FLOWS:
        source_key, factor, lag_seconds = _DERIVED_FLOWS[spec.key]
        base = factor * _profile(scenario, source_key, t_min - lag_seconds / 60.0)
    else:
        base = _profile(scenario, spec.key, t_min)
    seed = _stream_seed(scenario.seed, spec.key)
    wobble_amp, jitter_sigma = _wobble_noise_amplitudes(spec)
    return (
        base
        + _wobble(seed, t_min, wobble_amp)
        + jitter_sigma * _gaussian(seed, ks, 0x51CE)
    )


def _quality_series(scenario: Scenario, key: str, t_min: np.ndarray) -> np.ndarray:
    quality = np.full(t_min.shape, QUALITY_GOOD, dtype=object)
    for rule in scenario.quality_rules:
        if rule.sensor != key:
            continue
        end = np.inf if rule.end_min is None else rule.end_min
        mask = (t_min >= rule.start_min) & (t_min < end)
        quality[mask] = rule.kind
    return quality


def sensor_series(
    scenario: Scenario, spec: SensorSpec, ks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Values and quality for one sensor at the given sample indices."""
    t_min = _t_minutes(ks)
    values = _raw_values(scenario, spec, ks)
    quality = _quality_series(scenario, spec.key, t_min)

    # A stale sensor stops updating: freeze at the reading it stuck on.
    for rule in scenario.quality_rules:
        if rule.sensor != spec.key or rule.kind != QUALITY_STALE:
            continue
        end = np.inf if rule.end_min is None else rule.end_min
        mask = (t_min >= rule.start_min) & (t_min < end)
        if not mask.any():
            continue
        freeze_k = max(
            0,
            int(math.floor((rule.start_min + HISTORY_MINUTES) * 60.0 / SAMPLE_SECONDS)),
        )
        frozen_value = float(
            _raw_values(scenario, spec, np.array([freeze_k], dtype=np.int64))[0]
        )
        values = values.copy()
        values[mask] = frozen_value

    missing_mask = quality == QUALITY_MISSING
    if missing_mask.any():
        values = values.copy()
        values[missing_mask] = np.nan
    return values, quality


def generate_process_block(
    scenario: Scenario,
    specs: Sequence[SensorSpec],
    t0: pd.Timestamp,
    ks: np.ndarray,
) -> pd.DataFrame:
    """Long-format samples for all 16 sensors at the given sample indices."""
    timestamps = t0 + pd.to_timedelta(ks * SAMPLE_SECONDS, unit="s")
    frames: list[pd.DataFrame] = []
    for spec in specs:
        values, quality = sensor_series(scenario, spec, ks)
        alarms = [
            classify_alarm(value, spec.limits, sample_quality)
            for value, sample_quality in zip(values.tolist(), quality.tolist())
        ]
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_key": spec.key,
                    "pid_tag": spec.display_tag,
                    "display_name": spec.display_name,
                    "group_id": spec.group_id,
                    "group_name": spec.group_name,
                    "measurement": spec.measurement,
                    "value": values,
                    "unit": spec.unit,
                    "source": SOURCE_LOCAL,
                    "quality": quality,
                    "alarm_state": alarms,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def generate_electrical_block(
    scenario: Scenario, t0: pd.Timestamp, ks: np.ndarray
) -> pd.DataFrame:
    """Demo-only stack voltage/current (separate from the 16 P&ID sensors)."""
    timestamps = t0 + pd.to_timedelta(ks * SAMPLE_SECONDS, unit="s")
    t_min = _t_minutes(ks)
    columns: dict[str, object] = {"timestamp": timestamps}
    for pseudo_key, column in (
        ("STACK_VOLTAGE", "voltage_v"),
        ("STACK_CURRENT", "current_a"),
    ):
        baseline, wobble_amp, jitter_sigma = _ELECTRICAL_BASE[pseudo_key]
        seed = _stream_seed(scenario.seed, pseudo_key)
        columns[column] = (
            baseline
            + _wobble(seed, t_min, wobble_amp)
            + jitter_sigma * _gaussian(seed, ks, 0x0E1C)
        )
    return pd.DataFrame(columns)


def collect_events(
    block: pd.DataFrame,
    last_alarm: dict[str, str],
    events: list[dict[str, object]],
) -> None:
    """Append alarm-state transitions found in ``block`` to ``events``.

    Duplicate-safety: the caller feeds each sample exactly once (via the
    ``last_k`` pointer in ``advance_sim``), and an event is only recorded when
    a sample's state differs from the sensor's previously recorded state in
    ``last_alarm``. A five-second refresh with unchanged states therefore
    appends nothing, and re-rendering the page cannot re-append old events.
    """
    ordered = block.sort_values("timestamp", kind="mergesort")
    for key, group in ordered.groupby("sensor_key", sort=False):
        previous = last_alarm.get(key)
        for timestamp, state, value, unit, tag, name in zip(
            group["timestamp"],
            group["alarm_state"],
            group["value"],
            group["unit"],
            group["pid_tag"],
            group["display_name"],
        ):
            if previous is not None and state != previous:
                events.append(
                    {
                        "timestamp": timestamp,
                        "sensor_key": key,
                        "pid_tag": tag,
                        "sensor": name,
                        "previous": previous,
                        "new": state,
                        "value": value,
                        "unit": unit,
                        "source": SOURCE_LOCAL,
                    }
                )
            previous = state
        if previous is not None:
            last_alarm[key] = str(previous)


def _floor_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=DISPLAY_TZ).floor(f"{SAMPLE_SECONDS}s")


def ensure_sim(scenario_name: str, specs: Sequence[SensorSpec]) -> None:
    """Create (or cleanly regenerate) the demo session for a scenario.

    History is kept in ``st.session_state`` so ordinary widget-driven reruns
    reuse it unchanged; only a status change, the Reset readings button or a
    full page refresh starts a new session.
    """
    sim = st.session_state.get("sim")
    if sim is not None and sim["scenario"] == scenario_name:
        return

    scenario = SCENARIOS[scenario_name]
    t_gen = _floor_now()
    t0 = t_gen - pd.Timedelta(HISTORY_SECONDS, unit="s")
    ks = np.arange(HISTORY_SAMPLES, dtype=np.int64)

    frame = generate_process_block(scenario, specs, t0, ks)
    events: list[dict[str, object]] = []
    last_alarm: dict[str, str] = {}
    collect_events(frame, last_alarm, events)
    events.sort(key=lambda event: event["timestamp"])
    events = events[-EVENT_LIMIT:]

    st.session_state["sim"] = {
        "scenario": scenario_name,
        "t0": t0,
        "last_k": int(HISTORY_SAMPLES - 1),
        "frame": frame,
        "events": events,
        "last_alarm": last_alarm,
        "elec": generate_electrical_block(scenario, t0, ks),
    }
    advance_sim(specs)


def advance_sim(specs: Sequence[SensorSpec]) -> None:
    """Append only the genuinely new five-second samples up to now."""
    sim = st.session_state.get("sim")
    if sim is None:
        return
    scenario = SCENARIOS[sim["scenario"]]
    now = pd.Timestamp.now(tz=DISPLAY_TZ)
    target_k = int((now - sim["t0"]).total_seconds() // SAMPLE_SECONDS)
    if target_k <= sim["last_k"]:
        return

    ks = np.arange(sim["last_k"] + 1, target_k + 1, dtype=np.int64)
    block = generate_process_block(scenario, specs, sim["t0"], ks)
    collect_events(block, sim["last_alarm"], sim["events"])
    sim["events"].sort(key=lambda event: event["timestamp"])
    sim["events"] = sim["events"][-EVENT_LIMIT:]

    frame = pd.concat([sim["frame"], block], ignore_index=True)
    cutoff = frame["timestamp"].max() - pd.Timedelta(HISTORY_SECONDS, unit="s")
    sim["frame"] = frame.loc[frame["timestamp"] >= cutoff].reset_index(drop=True)

    elec_block = generate_electrical_block(scenario, sim["t0"], ks)
    elec = pd.concat([sim["elec"], elec_block], ignore_index=True)
    sim["elec"] = elec.loc[elec["timestamp"] >= cutoff].reset_index(drop=True)
    sim["last_k"] = target_k


def latest_snapshot(frame: pd.DataFrame, specs: Sequence[SensorSpec]) -> pd.DataFrame:
    """Latest sample per sensor, in registry presentation order."""
    latest = (
        frame.sort_values("timestamp", kind="mergesort")
        .groupby("sensor_key", sort=False)
        .tail(1)
        .set_index("sensor_key")
    )
    return latest.reindex([spec.key for spec in specs]).reset_index()


@dataclass(frozen=True)
class FeedStatus:
    label: str
    colour: str


def feed_status(frame: pd.DataFrame) -> FeedStatus:
    """Feed health from the age of the newest generated sample."""
    if frame.empty:
        return FeedStatus("Not updating", RED)
    newest = frame["timestamp"].max()
    age = (pd.Timestamp.now(tz=DISPLAY_TZ) - newest).total_seconds()
    if age <= ACTIVE_MAX_AGE_SECONDS:
        return FeedStatus("Updating", SECONDARY)
    if age <= DELAYED_MAX_AGE_SECONDS:
        return FeedStatus("Delayed", AMBER)
    return FeedStatus("Not updating", RED)


# ---------------------------------------------------------------------------
# 5. Formatting helpers
# ---------------------------------------------------------------------------


def format_number(value: object, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_timestamp(value: object, include_date: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(DISPLAY_TZ)
    if include_date:
        return timestamp.strftime("%d %b %Y at %H:%M:%S")
    return timestamp.strftime("%H:%M:%S")


_UI_DASH_TRANSLATION = str.maketrans({"-": " ", "–": " ", "—": " "})
_UI_TAG_TRANSLATION = str.maketrans("", "", "-–—")


def clean_ui_copy(text: object) -> str:
    """Remove dash separators from UI copy without changing numeric values."""
    return re.sub(r"\s+", " ", str(text).translate(_UI_DASH_TRANSLATION)).strip()


def clean_ui_tag(text: object) -> str:
    """Remove dash separators from displayed instrument tags."""
    return str(text).translate(_UI_TAG_TRANSLATION).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _rgba(hex_colour: str, alpha: float) -> str:
    hex_colour = hex_colour.lstrip("#")
    red_v, green_v, blue_v = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red_v},{green_v},{blue_v},{alpha})"


def logo_data_uri(path: Path) -> str:
    """Local logo as a data URI (no remote images); cached per session."""
    cached = st.session_state.get("_logo_uri")
    if cached is not None:
        return str(cached)
    try:
        raw = path.read_bytes()
    except OSError:
        raw = b""
    if raw:
        mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
        uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    else:
        uri = ""
    st.session_state["_logo_uri"] = uri
    return uri


@st.cache_data(show_spinner=False)
def _cached_image_data_uri(path_text: str, modified_ns: int) -> str:
    """Encode a local image; mtime participates in the cache key for easy updates."""
    del modified_ns  # used only to invalidate the cache when the file changes
    path = Path(path_text)
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if not raw:
        return ""
    suffix = path.suffix.lower()
    if suffix == ".png" or raw.startswith(b"\x89PNG"):
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def pid_image_data_uri() -> str:
    """Return the master P&ID as a responsive browser-safe data URI."""
    try:
        modified_ns = PID_IMAGE_FILE.stat().st_mtime_ns
    except OSError:
        return ""
    return _cached_image_data_uri(str(PID_IMAGE_FILE), modified_ns)


def _pid_marker_style(highlight: PidHighlight) -> str:
    """CSS for a marker whose centre is anchored to percentage coordinates."""
    common = (
        f"left:{highlight.x:.4f}%; top:{highlight.y:.4f}%; "
        "transform:translate(-50%,-50%);"
    )
    if highlight.shape == "rectangle":
        width = highlight.width if highlight.width is not None else 7.0
        height = highlight.height if highlight.height is not None else 4.0
        return common + f"width:{width:.4f}%; height:{height:.4f}%;"
    width = highlight.width if highlight.width is not None else DEFAULT_PID_CIRCLE_WIDTH
    return common + f"width:{width:.4f}%; aspect-ratio:1 / 1;"


def pid_overlay_html(spec: SensorSpec) -> str:
    """Full master P&ID with exactly one responsive yellow marker overlaid."""
    image_uri = pid_image_data_uri()
    if not image_uri:
        return ""
    marker_class = (
        "pid-marker pid-marker-rectangle"
        if spec.pid_highlight.shape == "rectangle"
        else "pid-marker pid-marker-circle"
    )
    marker_style = _pid_marker_style(spec.pid_highlight)
    alt_text = html.escape(
        f"NX-PID-001 with {spec.display_tag} {spec.display_name} highlighted",
        quote=True,
    )
    region_label = html.escape(
        f"Scrollable P&ID drawing locating {spec.display_tag}", quote=True
    )
    return (
        f'<div class="pid-image-shell" role="region" aria-label="{region_label}" '
        'tabindex="0">'
        '<div class="pid-image-wrap">'
        f'<img src="{image_uri}" alt="{alt_text}">'
        f'<span class="{marker_class}" style="{marker_style}" '
        'aria-hidden="true"></span>'
        '</div></div>'
    )


# ---------------------------------------------------------------------------
# 6. Theme and CSS
# ---------------------------------------------------------------------------

DARK_THEME = {
    "app_bg": "#0d1216",
    "text": "#f2f4f7",
    "subtext": "#a9b0b7",
    "sidebar_grad": "#151b20",
    "sidebar_border": "rgba(255,255,255,0.10)",
    "sidebar_text": "#ffffff",
    "sidebar_sub": "rgba(255,255,255,0.70)",
    "section_label": "rgba(255,255,255,0.62)",
    "card_bg": "transparent",
    "card_border": "rgba(255,255,255,0.10)",
    "input_bg": "#1c2329",
    "input_border": "rgba(255,255,255,0.16)",
    "popover_bg": "#1b222b",
    "panel_bg": "#171d22",
    "panel_border": "rgba(255,255,255,0.10)",
    "table_bg": "#171d22",
    "table_border": "rgba(255,255,255,0.08)",
    "chart_bg": "#171d22",
    "chart_border": "rgba(255,255,255,0.06)",
    "hero_bg": "transparent",
    "hero_border": "rgba(255,255,255,0.12)",
    "logo_filter": "none",
    "soft_bg": "rgba(255,255,255,0.04)",
    "divider": "rgba(255,255,255,0.12)",
    "utility_bg": "rgba(255,255,255,0.025)",
    "utility_border": "rgba(255,255,255,0.22)",
    "utility_icon": "#d7dde2",
    "utility_hover_bg": "#dcefae",
    "utility_hover_border": "#A7D730",
    "utility_hover_icon": "#1a2010",
    "tooltip_bg": "#ffffff",
    "tooltip_text": "#1a2010",
    "focus_ring": "#A7D730",
    "dialog_bg": "#171d22",
    "dialog_backdrop": "rgba(2,6,10,0.70)",
    "dialog_meta_bg": "#1c2329",
    "drawing_border": "#535c65",
    "drawing_shadow": "0 8px 24px rgba(0,0,0,0.28)",
}

LIGHT_THEME = {
    "app_bg": "#f4f6f3",
    "text": "#1a2010",
    "subtext": "#4a5240",
    "sidebar_grad": "#f8faf7",
    "sidebar_border": "rgba(73,152,35,0.20)",
    "sidebar_text": "#1a2010",
    "sidebar_sub": "#4a5240",
    "section_label": "#4a5240",
    "card_bg": "transparent",
    "card_border": "rgba(73,152,35,0.18)",
    "input_bg": "rgba(255,255,255,0.85)",
    "input_border": "rgba(73,152,35,0.30)",
    "popover_bg": "#ffffff",
    "panel_bg": "#ffffff",
    "panel_border": "rgba(73,152,35,0.15)",
    "table_bg": "#ffffff",
    "table_border": "rgba(73,152,35,0.15)",
    "chart_bg": "#ffffff",
    "chart_border": "rgba(73,152,35,0.14)",
    "hero_bg": "transparent",
    "hero_border": "rgba(73,152,35,0.20)",
    "logo_filter": "none",
    "soft_bg": "rgba(240,247,224,0.80)",
    "divider": "rgba(73,152,35,0.20)",
    "utility_bg": "rgba(48,52,60,0.025)",
    "utility_border": "rgba(48,52,60,0.25)",
    "utility_icon": "#59616a",
    "utility_hover_bg": "rgba(167,215,48,0.13)",
    "utility_hover_border": "rgba(73,152,35,0.55)",
    "utility_hover_icon": "#3f8122",
    "tooltip_bg": "#ffffff",
    "tooltip_text": "#1a2010",
    "focus_ring": "#499823",
    "dialog_bg": "#ffffff",
    "dialog_backdrop": "rgba(48,52,60,0.20)",
    "dialog_meta_bg": "#f7f9f5",
    "drawing_border": "#d5dbd2",
    "drawing_shadow": "0 6px 18px rgba(48,52,60,0.10)",
}

# Local font stack only: the app must remain fully usable offline, so no
# Google Fonts import (unlike app_v1). Hind is used when installed locally.
_CSS = Template("""
<style>
html, body, [class*="css"] { font-family: ${font_stack}; letter-spacing: 0; }
.stApp { background: ${app_bg}; color: ${text}; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stDecoration"], #MainMenu, footer {
    display: none !important; }
.block-container { max-width:1500px; padding:1rem clamp(1rem,2vw,2rem) 2rem; }
h1,h2,h3,h4,h5,h6 { color:${text} !important; font-weight:700; letter-spacing:0; }
p,span,label,li { color:${text}; }
.stCaption,.stMarkdown small { color:${subtext} !important; }

/* Streamlit help tooltips are rendered in a light portal outside the keyed
widget. Pin their foreground to a dark neutral in both app themes. */
[role="tooltip"] { background:${tooltip_bg} !important; color:${tooltip_text} !important; }
[role="tooltip"] * { color:${tooltip_text} !important; }

section[data-testid="stSidebar"] > div {
    background:${sidebar_grad}; border-right:1px solid ${sidebar_border}; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label { color:${sidebar_text} !important; }
section[data-testid="stSidebar"] hr { margin:.9rem 0; border-color:${divider}; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.55rem; }

.sidebar-title-card { padding:.3rem 0 .8rem; background:${card_bg};
    border-bottom:1px solid ${card_border}; margin-bottom:.35rem; }
.sidebar-kicker { color:${primary} !important; font-size:.76rem; font-weight:700;
    letter-spacing:.06em; text-transform:uppercase; margin:0 0 .2rem; }
.sidebar-title { color:${sidebar_text} !important; font-size:1.06rem; font-weight:700;
    line-height:1.15; margin:0; }
.sidebar-section-label { color:${section_label} !important; font-size:.74rem; font-weight:700;
    letter-spacing:.07em; text-transform:uppercase; margin:.4rem 0 .15rem; }
.sidebar-section-label.sidebar-section-break { margin:.75rem 0 .15rem; padding-top:.7rem;
    border-top:1px solid ${divider}; }
.st-key-reset_readings { margin-top:.1rem; }
.st-key-reset_readings button { background:transparent !important; color:${sidebar_sub} !important;
    border:1px solid ${input_border} !important; }
.st-key-reset_readings button:hover { color:${sidebar_text} !important;
    border-color:${primary} !important; }

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
.stSelectbox > div > div,
input[type="number"], .stTextInput input {
    background-color:${input_bg} !important; border-color:${input_border} !important;
    color:${text} !important; }
.stSelectbox input { color:${text} !important; -webkit-text-fill-color:${text} !important; }
.stSelectbox button { color:${text} !important; }
div[data-baseweb="select"] span, div[data-baseweb="input"] input,
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stSelectboxVirtualDropdown"] span { color:${text} !important; }
ul[data-testid="stSelectboxVirtualDropdown"] { background:${popover_bg} !important; }

.stButton > button,.stDownloadButton > button { font-weight:600; border-radius:8px; }

/* The app's own theme switch is independent of Streamlit's chrome theme.
Scope the dialog surface explicitly so dark mode never renders light text
on Streamlit's default white modal. */
[data-testid="stDialog"] { background:${dialog_backdrop}; }
[data-testid="stDialog"] > div:first-child {
    background:${dialog_bg}; border:1px solid ${panel_border};
    box-shadow:0 18px 52px rgba(0,0,0,.30); }
[data-testid="stDialog"] [role="dialog"] { color:${text}; }
[data-testid="stDialog"] [role="dialog"] > h2 { color:${text} !important; }
[data-testid="stDialog"] button[aria-label="Close"] {
    color:${subtext}; border-radius:6px; }
[data-testid="stDialog"] button[aria-label="Close"]:hover {
    color:${text}; background:${soft_bg}; }
[data-testid="stDialog"] button[aria-label="Close"]:focus-visible {
    outline:2px solid ${focus_ring}; outline-offset:2px; }

.hero-banner { display:flex; justify-content:space-between; align-items:center; gap:1.1rem;
    padding:.15rem 0 .85rem; border-bottom:1px solid ${hero_border};
    background:${hero_bg}; margin-bottom:.85rem; }
.hero-copy { max-width:74%; min-width:0; }
.hero-kicker { display:block; color:${primary} !important; font-size:.74rem; font-weight:700;
    letter-spacing:.06em; text-transform:uppercase; margin:0 0 .18rem; }
.hero-title { margin:0; color:${text} !important; font-size:34px !important;
    line-height:1.12; font-weight:700; }
.hero-sub { margin:.22rem 0 0; color:${subtext} !important; font-size:.92rem; font-weight:500; }
.hero-side { display:flex; align-items:center; justify-content:flex-end; gap:1rem; flex:0 0 auto; }
.hero-status { display:flex; align-items:center; gap:.55rem; min-width:9rem; }
.hero-status-dot { width:.68rem; height:.68rem; border-radius:50%; flex:0 0 auto; }
.hero-status-copy { display:flex; flex-direction:column; gap:.05rem; }
.hero-status-label { color:${text} !important; font-size:.84rem; font-weight:700; line-height:1.25; }
.hero-status-meta { color:${subtext} !important; font-size:.75rem; line-height:1.25; }
.hero-logos { display:flex; align-items:center; justify-content:flex-end; flex:0 0 auto; }
.hero-logos img { height:68px; width:auto; object-fit:contain; filter:${logo_filter}; }
.logo-fallback { color:${primary} !important; font-size:1.5rem; font-weight:800;
    letter-spacing:.02em; }

.stack-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.7rem; margin:0 0 1rem; }
.stack-card { position:relative; overflow:hidden; min-width:0;
    background:linear-gradient(135deg,rgba(167,215,48,.12) 0%,
        rgba(78,168,222,.05) 48%,transparent 80%),${panel_bg};
    border:1px solid ${panel_border}; border-radius:8px; padding:.75rem .9rem;
    min-height:6.4rem; box-shadow:0 7px 20px rgba(0,0,0,.10); }
.stack-card::before { content:""; position:absolute; inset:0 0 auto; height:2px;
    background:linear-gradient(90deg,${primary},${water_blue}); opacity:.85; }
.stack-icon { position:absolute; top:.62rem; right:.72rem; display:grid;
    place-items:center; width:2rem; height:2rem; border-radius:7px;
    background:rgba(167,215,48,.10); border:1px solid rgba(78,168,222,.18);
    color:${primary} !important; font-family:"Material Symbols Rounded";
    font-size:1.2rem; font-weight:400; font-style:normal; line-height:1;
    font-variation-settings:"FILL" 0,"wght" 500,"GRAD" 0,"opsz" 24; }
.stack-label { display:block; color:${subtext} !important; font-size:.7rem; font-weight:700;
    letter-spacing:.05em; text-transform:uppercase; margin-bottom:.28rem;
    padding-right:2.35rem; overflow-wrap:anywhere; }
.stack-value { color:${text} !important; font-size:1.16rem; font-weight:700; line-height:1.15; }
.stack-muted { color:${subtext} !important; font-size:.98rem; font-weight:600; }
.stack-unit { margin-left:.25rem; color:${subtext} !important; font-size:.78rem; font-weight:600; }
.stack-sub { margin-top:.3rem; color:${subtext} !important; font-size:.75rem; line-height:1.35;
    display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; }
.section-heading { display:flex; align-items:baseline; justify-content:space-between;
    gap:.8rem; margin:0 0 .45rem; }
.section-heading h2 { margin:0 !important; padding:0 !important; color:${text} !important;
    font-size:1rem !important; line-height:1.3; font-weight:700; }

.rail { display:flex; flex-direction:column; gap:0; overflow:hidden; background:${panel_bg};
    border:1px solid ${panel_border}; border-radius:8px; }
.rail-item { background:transparent; border:0; border-bottom:1px solid ${divider};
    padding:.65rem .75rem; border-left:3px solid ${light_grey}; }
.rail-item:last-child { border-bottom:0; }
.rail-head { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.rail-name { color:${text} !important; font-size:.82rem; font-weight:700; }
.rail-status { font-size:.68rem; font-weight:700; padding:.08rem .5rem; border-radius:999px;
    white-space:nowrap; }
.rail-line { margin-top:.18rem; color:${subtext} !important; font-size:.74rem; line-height:1.35; }
.rail-quality { margin-top:.18rem; color:${amber} !important; font-size:.72rem;
    font-weight:600; line-height:1.3; }

.crit-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; }
.crit-card { min-width:0; background:${panel_bg}; border:1px solid ${panel_border};
    border-radius:8px; padding:.7rem .8rem; border-left:3px solid ${light_grey};
    min-height:7.9rem; }
.crit-top { display:flex; align-items:center; justify-content:space-between; gap:.32rem;
    margin-bottom:.22rem; }
.crit-tag { color:${subtext} !important; font-size:.72rem; font-weight:700;
    letter-spacing:.05em; }
.alarm-badge { font-size:.66rem; font-weight:700; padding:.08rem .5rem; border-radius:999px;
    white-space:nowrap; }
.crit-name { color:${text} !important; font-size:.83rem; font-weight:600; line-height:1.25;
    min-height:1.8rem; overflow-wrap:anywhere; }
.crit-note { color:${subtext} !important; font-size:.7rem; line-height:1.3; margin-top:.1rem; }
.crit-value { margin-top:.25rem; color:${text} !important; font-size:1.28rem; font-weight:700;
    line-height:1.1; padding-right:2.65rem; }
.crit-unit { margin-left:.25rem; color:${subtext} !important; font-size:.8rem; font-weight:600; }
.crit-missing { margin-top:.3rem; padding-right:2.65rem; color:${subtext} !important;
    font-size:.98rem; font-weight:600; }
.crit-meta { margin-top:.4rem; display:flex; align-items:center; gap:.4rem; flex-wrap:wrap;
    padding-right:2.65rem; color:${subtext} !important; font-size:.7rem; }
.qual-chip { display:inline-block; padding:.04rem .42rem; border-radius:6px;
    font-size:.62rem; font-weight:700; letter-spacing:.04em; }
.qual-good { border:1px solid ${secondary}; color:${secondary} !important;
    background:transparent; }
.qual-bad { background:${light_grey}; color:#ffffff !important; }

/* Native Streamlit containers are used for key sensor cards so each card can
contain a real, keyboard-accessible P&ID locator button. */
[class*="st-key-critical_card_"] {
    position:relative; min-height:8.35rem; background:${panel_bg};
    border:1px solid ${panel_border}; border-left:3px solid ${light_grey};
    border-radius:8px; padding:.68rem .78rem; box-sizing:border-box;
}
[class*="st-key-critical_card_"] [data-testid="stVerticalBlock"] { gap:.08rem !important; }
[class*="st-key-pid_help_"] {
    position:absolute; right:.5rem; bottom:.5rem; z-index:4;
    width:2.125rem; height:2.125rem; }
[class*="st-key-pid_help_"] button {
    width:2.125rem !important; min-width:2.125rem !important;
    height:2.125rem !important; min-height:2.125rem !important;
    padding:0 !important; border-radius:6px !important;
    border:1px solid ${utility_border} !important;
    background:${utility_bg} !important; color:${utility_icon} !important;
    line-height:1 !important; box-shadow:none !important;
    transition:background-color .14s ease,border-color .14s ease,color .14s ease;
}
[class*="st-key-pid_help_"] button:hover {
    border-color:${utility_hover_border} !important;
    color:${utility_hover_icon} !important;
    background:${utility_hover_bg} !important;
}
[class*="st-key-pid_help_"] button:focus-visible {
    outline:2px solid ${focus_ring} !important; outline-offset:2px !important; }
[class*="st-key-pid_help_"] [data-testid="stIconMaterial"] {
    color:inherit !important; font-size:1.08rem !important;
    font-variation-settings:"FILL" 0,"wght" 450,"GRAD" 0,"opsz" 20; }
/* Keep a descriptive accessible name while presenting only the native icon. */
[class*="st-key-pid_help_"] [data-testid="stMarkdownContainer"] {
    position:absolute !important; width:1px !important; height:1px !important;
    padding:0 !important; margin:-1px !important; overflow:hidden !important;
    clip:rect(0,0,0,0) !important; white-space:nowrap !important; border:0 !important; }

.pid-dialog-title { margin:0 0 .22rem !important; padding:0 !important;
    color:${text} !important; line-height:1.25; }
.pid-dialog-tag { display:block; color:${subtext} !important; font-size:.72rem;
    font-weight:750; letter-spacing:.07em; line-height:1.25; text-transform:uppercase; }
.pid-dialog-name { display:block; margin-top:.08rem; color:${text} !important;
    font-size:1.08rem; font-weight:700; line-height:1.3; }
.pid-dialog-description { margin:0 0 .62rem; color:${subtext} !important;
    font-size:.86rem; line-height:1.45; max-width:72rem; }
.pid-meta-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.55rem; margin:.08rem 0 .72rem; }
.pid-meta-card { background:${dialog_meta_bg}; border:1px solid ${panel_border};
    border-radius:6px; padding:.52rem .65rem; min-width:0; }
.pid-meta-label { display:block; color:${subtext} !important; font-size:.65rem;
    font-weight:700; letter-spacing:.05em; text-transform:uppercase; margin-bottom:.14rem; }
.pid-meta-value { color:${text} !important; font-size:.86rem; font-weight:700;
    line-height:1.25; overflow-wrap:anywhere; }
.pid-state-badge { display:inline-block; width:max-content; padding:.08rem .48rem;
    border-radius:999px; font-size:.72rem; font-weight:750; line-height:1.35; }
.pid-drawing-heading { display:flex; align-items:baseline; justify-content:space-between;
    gap:.7rem; margin:.15rem 0 .38rem; padding-top:.62rem;
    border-top:1px solid ${divider}; }
.pid-drawing-title { color:${text} !important; font-size:.76rem; font-weight:750;
    letter-spacing:.055em; text-transform:uppercase; }
.pid-drawing-id { color:${subtext} !important; font-size:.72rem; font-weight:650; }
.pid-image-shell { width:100%; overflow:auto; background:#ffffff;
    border:1px solid ${drawing_border}; border-radius:7px; padding:.42rem;
    box-shadow:${drawing_shadow}; box-sizing:border-box; scrollbar-gutter:stable; }
.pid-image-shell:focus-visible { outline:2px solid ${focus_ring}; outline-offset:2px; }
.pid-image-wrap { position:relative; width:100%; min-width:760px; line-height:0; }
.pid-image-wrap img { display:block; width:100%; height:auto; user-select:none; }
.pid-marker { position:absolute; box-sizing:border-box; border:3px solid #FFD400;
    background:rgba(255,212,0,.24); box-shadow:0 0 0 1.5px rgba(48,52,60,.68);
    pointer-events:none; }
.pid-marker-circle { border-radius:50%; }
.pid-marker-rectangle { border-radius:3px; }
.pid-quality-note { margin:.5rem 0 0; color:${subtext} !important; font-size:.74rem;
    line-height:1.4; }

.st-key-pid_finder_tool { margin:.05rem 0 .58rem; }
.st-key-pid_finder_tool [data-testid="stVerticalBlock"] { gap:.15rem; }
.st-key-pid_sensor_finder_button button {
    min-height:2.5rem; border-radius:7px !important;
    border:1px solid ${input_border} !important; background:${utility_bg} !important;
    color:${text} !important; box-shadow:none !important; }
.st-key-pid_sensor_finder_button button:hover {
    border-color:${utility_hover_border} !important;
    background:${utility_hover_bg} !important; color:${utility_hover_icon} !important; }
.st-key-pid_sensor_finder_button button:hover p,
.st-key-pid_sensor_finder_button button:hover span {
    color:${utility_hover_icon} !important; }
.st-key-pid_sensor_finder_button button:focus-visible {
    outline:2px solid ${focus_ring} !important; outline-offset:2px !important; }

.legend-row { display:flex; align-items:flex-start; gap:.45rem; margin:.15rem 0; }
.legend-dot { width:.7rem; height:.7rem; border-radius:4px; flex:0 0 auto; margin-top:.18rem; }
.legend-text { color:${sidebar_text} !important; font-size:.78rem; line-height:1.35; }

.note-inline { margin:.35rem 0 0; color:${subtext} !important; font-size:.76rem;
    line-height:1.45; }
.section-title { margin:1.0rem 0 .55rem; color:${text} !important;
    font-size:1.15rem !important; font-weight:700; line-height:1.25; padding:0 !important; }
.table-note { margin:.5rem 0 0; color:${subtext} !important; font-size:.78rem; line-height:1.4; }

.stPlotlyChart { min-height:380px; margin-bottom:.7rem; clear:both; overflow:hidden;
    background:${chart_bg}; border:1px solid ${chart_border}; border-radius:8px;
    padding:.5rem .9rem .2rem .45rem; box-sizing:border-box; }
.data-table-shell { width:100%; overflow:auto; scrollbar-gutter:stable;
    background:${table_bg}; border:1px solid ${table_border}; border-radius:8px;
    box-sizing:border-box; }
.data-table-shell:focus-visible { outline:2px solid ${water_blue}; outline-offset:2px; }
.data-table { width:100%; border-collapse:separate; border-spacing:0;
    font-size:.82rem; line-height:1.3; }
.data-table th { position:sticky; top:0; z-index:2; padding:.62rem .7rem;
    background:${input_bg}; color:${subtext}; border-right:1px solid ${table_border};
    border-bottom:1px solid ${table_border}; font-weight:600; text-align:left;
    white-space:nowrap; }
.data-table td { padding:.58rem .7rem; background:${table_bg}; color:${text};
    border-right:1px solid ${table_border}; border-bottom:1px solid ${table_border};
    vertical-align:middle; white-space:nowrap; }
.data-table th:last-child,.data-table td:last-child { border-right:0; }
.data-table tbody tr:last-child td { border-bottom:0; }
.data-table tbody tr:hover td { filter:brightness(1.04); }

/* Streamlit's fixed st.columns rows do not wrap at laptop widths. The keyed
wrappers keep these overrides local to the monitoring and locator tools. */
@media (max-width:1250px) {
    .crit-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .st-key-critical_sensor_grid [data-testid="stHorizontalBlock"] {
        flex-wrap:wrap !important; }
    .st-key-critical_sensor_grid [data-testid="stColumn"] {
        flex:1 1 calc(50% - .75rem) !important;
        width:calc(50% - .75rem) !important;
        max-width:calc(50% - .75rem) !important; }
}
@media (max-width:1080px) {
    .hero-banner { flex-direction:column; align-items:flex-start; }
    .hero-copy { max-width:100%; }
    .hero-side,.hero-logos { justify-content:flex-start; }
    .hero-logos img { height:64px; }
}
@media (max-width:800px) {
    .st-key-monitoring_overview [data-testid="stHorizontalBlock"]:has(.st-key-monitoring_safety):has(.st-key-monitoring_sensors) {
        flex-direction:column !important; gap:.9rem !important; }
    .st-key-monitoring_overview [data-testid="stHorizontalBlock"]:has(.st-key-monitoring_safety):has(.st-key-monitoring_sensors) > [data-testid="stColumn"] {
        flex:1 1 100% !important; width:100% !important; max-width:100% !important; }
}
@media (max-width:760px) {
    .block-container { padding:3.4rem .8rem 1.5rem; }
    .hero-banner { padding:.2rem 0 .75rem; }
    .hero-title { font-size:28px !important; }
    .hero-side { width:100%; justify-content:space-between; }
    .stack-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:.55rem; }
    .stack-card { min-height:5.8rem; padding:.65rem .7rem; }
    .stack-card:last-child { grid-column:1 / -1; }
    .crit-grid { grid-template-columns:1fr; gap:.55rem; }
    .pid-meta-grid { grid-template-columns:1fr; }
    .pid-image-wrap { min-width:680px; }
    .st-key-pid_finder_tool [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
    .st-key-pid_finder_tool [data-testid="stColumn"] {
        flex:1 1 100% !important; width:100% !important; max-width:100% !important; }
    .stPlotlyChart { min-height:340px; }
}
@media (max-width:520px) {
    .st-key-critical_sensor_grid [data-testid="stColumn"] {
        flex:1 1 100% !important; width:100% !important; max-width:100% !important; }
    [class*="st-key-critical_card_"] { min-height:7.8rem; }
    .pid-image-wrap { min-width:620px; }
}
</style>
""")


def is_light() -> bool:
    return st.session_state.get("theme", "dark") == "light"


def inject_css(theme: str) -> None:
    tokens = dict(LIGHT_THEME if theme == "light" else DARK_THEME)
    tokens.update(
        {
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "amber": AMBER,
            "red": RED,
            "light_grey": LIGHT_GREY,
            "water_blue": WATER_BLUE,
            "font_stack": FONT_STACK,
        }
    )
    st.markdown(_CSS.substitute(tokens), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 7. Reusable HTML fragments (all user-editable text is escaped)
# ---------------------------------------------------------------------------


def _alarm_badge(state: str) -> str:
    background, foreground = state_badge_colours(state)
    return (
        f'<span class="alarm-badge" style="background:{background}; '
        f'color:{foreground};">{html.escape(alarm_display_text(state))}</span>'
    )


def _quality_chip(quality: str) -> str:
    css = "qual-good" if quality == QUALITY_GOOD else "qual-bad"
    return f'<span class="qual-chip {css}">{html.escape(quality.title())}</span>'


def _pid_reading_text(spec: SensorSpec, row: pd.Series) -> str:
    quality = str(row["quality"])
    value = row["value"]
    if quality == QUALITY_MISSING or pd.isna(value):
        return "No reading"
    if quality == QUALITY_STALE:
        return "Stale reading"
    if quality == QUALITY_BAD:
        return "Invalid reading"
    return f"{format_number(value, spec.decimals)} {spec.unit}"


def request_pid_dialog(sensor_key: str) -> None:
    """Keep the requested P&ID dialog open across five-second fragment refreshes."""
    st.session_state["pid_dialog_sensor"] = sensor_key


def clear_pid_dialog() -> None:
    st.session_state.pop("pid_dialog_sensor", None)


@st.dialog(
    "Sensor location on P&ID",
    width="large",
    icon=":material/account_tree:",
    on_dismiss=clear_pid_dialog,
)
def show_pid_dialog(spec: SensorSpec, row: pd.Series) -> None:
    """Show one sensor in plant context without generating duplicate images."""
    state = str(row["alarm_state"])
    quality = str(row["quality"])
    reading = _pid_reading_text(spec, row)
    state_colour, state_foreground = state_badge_colours(state)

    st.markdown(
        '<h3 class="pid-dialog-title">'
        f'<span class="pid-dialog-tag">{html.escape(spec.display_tag)}</span>'
        f'<span class="pid-dialog-name">{html.escape(spec.display_name)}</span>'
        '</h3>'
        f'<p class="pid-dialog-description">{html.escape(spec.description)}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pid-meta-grid">'
        '<div class="pid-meta-card"><span class="pid-meta-label">Current reading</span>'
        f'<span class="pid-meta-value">{html.escape(reading)}</span></div>'
        '<div class="pid-meta-card"><span class="pid-meta-label">Alarm state</span>'
        f'<span class="pid-state-badge" style="background:{state_colour};'
        f'color:{state_foreground};">{html.escape(alarm_display_text(state))}</span></div>'
        '<div class="pid-meta-card"><span class="pid-meta-label">System area</span>'
        f'<span class="pid-meta-value">{html.escape(spec.group_name)}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not PID_IMAGE_FILE.exists():
        st.error(
            "The master P&ID image is missing. Expected: "
            "assets/pid/NX-PID-001.jpg"
        )
        return

    overlay = pid_overlay_html(spec)
    if not overlay:
        st.error("The master P&ID image could not be read.")
        return
    st.markdown(
        '<div class="pid-drawing-heading">'
        '<span class="pid-drawing-title">Full P&amp;ID</span>'
        '<span class="pid-drawing-id">NX-PID-001</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.html(overlay)

    if quality != QUALITY_GOOD:
        st.markdown(
            f'<p class="pid-quality-note">Data quality: '
            f'{html.escape(quality.title())}</p>',
            unsafe_allow_html=True,
        )


def render_header(snapshot: pd.DataFrame, frame: pd.DataFrame) -> None:
    feed = feed_status(frame)
    overall = worst_state(snapshot["alarm_state"])
    overall_colour, _ = state_badge_colours(overall)
    if overall in CRITICAL_STATES:
        overall_text = "Critical alarm"
    elif overall in WARNING_STATES:
        overall_text = "Warning active"
    else:
        overall_text = SEVERITY_WORD[overall]
    updated = format_timestamp(frame["timestamp"].max())
    status_meta = f"Updated {updated}"
    if feed.label != "Updating":
        status_meta = f"{feed.label}. {status_meta}"

    logo_uri = logo_data_uri(LOGO_FILE)
    if logo_uri:
        logo_html = f'<img src="{logo_uri}" alt="HydroStar">'
    else:
        logo_html = '<span class="logo-fallback">HydroStar</span>'

    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-copy">'
        '<span class="hero-kicker">HydroStar</span>'
        f'<h1 class="hero-title">{html.escape(PAGE_TITLE)}</h1>'
        f'<p class="hero-sub">{html.escape(PAGE_SUBTITLE)}</p>'
        "</div>"
        '<div class="hero-side">'
        '<div class="hero-status">'
        f'<span class="hero-status-dot" style="background:{overall_colour};"></span>'
        '<span class="hero-status-copy">'
        f'<span class="hero-status-label">{html.escape(overall_text)}</span>'
        f'<span class="hero-status-meta">{html.escape(status_meta)}</span>'
        "</span></div>"
        f'<div class="hero-logos">{logo_html}</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _stat_card(
    label: str, value_html: str, sub_html: str = "", icon: str = ""
) -> str:
    sub_block = f'<div class="stack-sub">{sub_html}</div>' if sub_html else ""
    icon_block = (
        f'<span class="stack-icon" aria-hidden="true">{html.escape(icon)}</span>'
        if icon
        else ""
    )
    return (
        '<div class="stack-card">'
        f"{icon_block}"
        f'<span class="stack-label">{html.escape(label)}</span>'
        f"<div>{value_html}</div>{sub_block}</div>"
    )


def render_stack_row(
    elec: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> None:
    latest_elec = elec.iloc[-1]
    voltage = float(latest_elec["voltage_v"])
    current = float(latest_elec["current_a"])
    # Power rule: never from voltage alone — always voltage x current / 1000.
    power_kw = voltage * current / 1000.0

    indexed = snapshot.set_index("sensor_key")
    hydrogen_row = indexed.loc["FT001_H2"]
    oxygen_row = indexed.loc["FT002_O2"]
    hydrogen_value = hydrogen_row["value"]
    hydrogen_quality = str(hydrogen_row["quality"])
    if hydrogen_quality in {QUALITY_MISSING, QUALITY_BAD} or pd.isna(hydrogen_value):
        hydrogen_html = '<span class="stack-muted">No reading</span>'
    elif hydrogen_quality == QUALITY_STALE:
        hydrogen_html = '<span class="stack-muted">Stale reading</span>'
    else:
        hydrogen_html = (
            f'<span class="stack-value">{format_number(hydrogen_value, 2)}</span>'
            '<span class="stack-unit">Nm³/h</span>'
        )

    oxygen_value = oxygen_row["value"]
    oxygen_quality = str(oxygen_row["quality"])
    ratio_available = (
        hydrogen_quality == QUALITY_GOOD
        and oxygen_quality == QUALITY_GOOD
        and not pd.isna(hydrogen_value)
        and not pd.isna(oxygen_value)
        and float(oxygen_value) > 0.0
    )
    if ratio_available:
        product_ratio = float(hydrogen_value) / float(oxygen_value)
        ratio_html = f'<span class="stack-value">{product_ratio:.2f} : 1</span>'
        ratio_sub = "H₂ to O₂ flow ratio"
    else:
        ratio_html = '<span class="stack-muted">Unavailable</span>'
        ratio_sub = "Requires valid FT001 and FT002 readings"

    cards = [
        _stat_card(
            "Stack power input",
            f'<span class="stack-value">{power_kw:,.2f}</span>'
            '<span class="stack-unit">kW</span>',
            f"{voltage:,.1f} V × {current:,.1f} A",
            "electric_bolt",
        ),
        _stat_card(
            "Hydrogen production rate",
            hydrogen_html,
            "FT001",
            "air",
        ),
        _stat_card(
            "Product gas balance",
            ratio_html,
            ratio_sub,
            "balance",
        ),
    ]
    st.markdown(f'<div class="stack-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _rail_item_html(
    name: str,
    state: str,
    status_text: str,
    detail: str,
    quality_note: str = "",
) -> str:
    colour, foreground = state_badge_colours(state)
    quality_html = (
        f'<div class="rail-quality">{html.escape(quality_note)}</div>'
        if quality_note
        else ""
    )
    return (
        f'<div class="rail-item" style="border-left-color:{colour};">'
        '<div class="rail-head">'
        f'<span class="rail-name">{html.escape(name)}</span>'
        f'<span class="rail-status" style="background:{colour}; color:{foreground};">'
        f"{html.escape(status_text)}</span>"
        "</div>"
        f'<div class="rail-line">{html.escape(detail)}</div>'
        f"{quality_html}</div>"
    )


def _quality_note(members: pd.DataFrame) -> str:
    stale = int((members["quality"] == QUALITY_STALE).sum())
    missing = int((members["quality"] == QUALITY_MISSING).sum())
    bad = int((members["quality"] == QUALITY_BAD).sum())
    parts = []
    if stale:
        parts.append(f"{stale} stale reading" + ("s" if stale > 1 else ""))
    if missing:
        parts.append(f"{missing} missing reading" + ("s" if missing > 1 else ""))
    if bad:
        parts.append(f"{bad} bad reading" + ("s" if bad > 1 else ""))
    return "Data quality: " + ", ".join(parts) if parts else ""


def _sensor_labels(rows: pd.DataFrame) -> list[str]:
    """Return compact P&ID labels for a safety summary."""
    return [clean_ui_tag(row["pid_tag"]) for _, row in rows.iterrows()]


def render_safety_rail(snapshot: pd.DataFrame) -> None:
    indexed = snapshot.set_index("sensor_key")
    items: list[str] = []

    for name, member_keys in SAFETY_ITEMS:
        members = indexed.loc[list(member_keys)]
        state = worst_state(members["alarm_state"].tolist())
        if state == STATE_NORMAL:
            detail = ""
        elif state == STATE_UNCONFIGURED:
            detail = "Limits require approval"
        elif state == STATE_UNKNOWN:
            unknown_rows = members.loc[members["alarm_state"] == STATE_UNKNOWN]
            detail = f"{', '.join(_sensor_labels(unknown_rows))} unavailable"
        else:
            worst_rows = members.loc[members["alarm_state"] == state]
            labels = ", ".join(_sensor_labels(worst_rows))
            detail = f"{labels} {alarm_display_text(state)}"
        items.append(
            _rail_item_html(
                name, state, SEVERITY_WORD[state], detail, _quality_note(members)
            )
        )

    st.markdown(
        '<div class="section-heading"><h2>Safety overview</h2></div>'
        f'<div class="rail">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def _critical_card_html(spec: SensorSpec, row: pd.Series) -> str:
    """Card body; the outer card is a keyed Streamlit container."""
    state = str(row["alarm_state"])
    quality = str(row["quality"])
    value = row["value"]

    if quality == QUALITY_MISSING or pd.isna(value):
        value_html = '<div class="crit-missing">No reading</div>'
    elif quality == QUALITY_STALE:
        value_html = '<div class="crit-missing">Stale reading</div>'
    else:
        value_html = (
            '<div class="crit-value">'
            f"{format_number(value, spec.decimals)}"
            f'<span class="crit-unit">{html.escape(spec.unit)}</span></div>'
        )

    quality_html = (
        f'<div class="crit-meta">{_quality_chip(quality)}</div>'
        if quality != QUALITY_GOOD
        else ""
    )
    return (
        '<div class="crit-top">'
        f'<span class="crit-tag">{html.escape(spec.display_tag)}</span>'
        f"{_alarm_badge(state)}"
        "</div>"
        f'<div class="crit-name">{html.escape(spec.display_name)}</div>'
        f"{value_html}"
        f"{quality_html}"
    )


def render_critical_cards(
    snapshot: pd.DataFrame, specs_map: Mapping[str, SensorSpec]
) -> None:
    """Eight priority sensor cards with native, accessible P&ID locators."""
    indexed = snapshot.set_index("sensor_key")
    st.markdown(
        '<div class="section-heading"><h2>Key sensors</h2></div>',
        unsafe_allow_html=True,
    )

    # Apply each live alarm colour to the left edge of its keyed container.
    style_rules = []
    for key in CRITICAL_CARD_ORDER:
        colour, _ = state_badge_colours(str(indexed.loc[key]["alarm_state"]))
        style_rules.append(
            f".st-key-critical_card_{key}{{border-left-color:{colour}!important;}}"
        )
    st.markdown(f"<style>{''.join(style_rules)}</style>", unsafe_allow_html=True)

    with st.container(key="critical_sensor_grid"):
        for row_start in range(0, len(CRITICAL_CARD_ORDER), 4):
            columns = st.columns(4, gap="small")
            for column, key in zip(
                columns, CRITICAL_CARD_ORDER[row_start : row_start + 4]
            ):
                spec = specs_map[key]
                row = indexed.loc[key]
                with column:
                    with st.container(key=f"critical_card_{key}", border=False):
                        st.markdown(
                            _critical_card_html(spec, row), unsafe_allow_html=True
                        )
                        st.button(
                            f"Locate {spec.display_tag} on P&ID",
                            key=f"pid_help_{key}",
                            help=f"Locate {spec.display_tag} on P&ID",
                            icon=":material/location_searching:",
                            type="tertiary",
                            width=34,
                            on_click=request_pid_dialog,
                            args=(key,),
                        )


# ---------------------------------------------------------------------------
# 8. All Sensors table
# ---------------------------------------------------------------------------

TABLE_COLUMNS = ["P&ID tag", "Description", "Reading", "Alarm"]


def build_sensor_table(
    snapshot: pd.DataFrame, specs: Sequence[SensorSpec]
) -> tuple[pd.DataFrame, list[str]]:
    indexed = snapshot.set_index("sensor_key")
    rows: list[dict[str, str]] = []
    states: list[str] = []
    for spec in specs:
        row = indexed.loc[spec.key]
        state = str(row["alarm_state"])
        quality = str(row["quality"])
        value = row["value"]
        if quality == QUALITY_MISSING or pd.isna(value):
            value_text = "No reading"
        elif quality == QUALITY_STALE:
            value_text = "Stale reading"
        else:
            value_text = f"{format_number(value, spec.decimals)} {spec.unit}"
        rows.append(
            {
                "P&ID tag": spec.display_tag,
                "Description": spec.description,
                "Reading": value_text,
                "Alarm": alarm_display_text(state),
            }
        )
        states.append(state)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), states


_AMBER_TEXT = "#B97B00"  # readable amber-family text on light backgrounds


def _data_table_html(
    table: pd.DataFrame,
    cell_styles: pd.DataFrame,
    *,
    aria_label: str,
    max_height: int,
    min_width: int,
) -> str:
    """Render a bounded semantic table with escaped values and fixed theming."""
    if cell_styles.shape != table.shape:
        raise ValueError("Table values and cell styles must have matching shapes")

    headers = "".join(
        f'<th scope="col">{html.escape(str(column))}</th>'
        for column in table.columns
    )
    rows = []
    for row_index, values in enumerate(table.itertuples(index=False, name=None)):
        cells = []
        for column_index, value in enumerate(values):
            style = str(cell_styles.iat[row_index, column_index]).strip()
            style_attr = (
                f' style="{html.escape(style, quote=True)}"' if style else ""
            )
            cells.append(
                f"<td{style_attr}>{html.escape(str(value))}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    safe_label = html.escape(aria_label, quote=True)
    return (
        f'<div class="data-table-shell" role="region" aria-label="{safe_label}" '
        f'tabindex="0" style="max-height:{int(max_height)}px;">'
        f'<table class="data-table" aria-label="{safe_label}" '
        f'style="min-width:{int(min_width)}px;">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def sensor_table_styles(table: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    """Keep descriptions readable and colour value/alarm cells by state."""
    light_theme = is_light()
    base_background = "#ffffff" if light_theme else "#171d22"
    base_text = "#1a2010" if light_theme else "#f2f4f7"
    css = pd.DataFrame(
        f"background-color:{base_background}; color:{base_text};",
        index=table.index,
        columns=table.columns,
    )
    description_col = table.columns.get_loc("Description")
    value_col = table.columns.get_loc("Reading")
    alarm_col = table.columns.get_loc("Alarm")
    for i, state in enumerate(states):
        background, foreground = state_badge_colours(state)
        text_colour = _AMBER_TEXT if light_theme and background == AMBER else background
        value_background = _rgba(background, 0.14) if light_theme else base_background
        css.iloc[i, description_col] = (
            f"background-color:{base_background}; color:{base_text}; "
            "white-space:normal; min-width:390px; line-height:1.4;"
        )
        css.iloc[i, value_col] = (
            f"background-color:{value_background}; color:{text_colour}; "
            "font-weight:700;"
        )
        css.iloc[i, alarm_col] = (
            f"background-color:{background}; color:{foreground}; font-weight:600;"
        )
    return css


def snapshot_csv(snapshot: pd.DataFrame, specs: Sequence[SensorSpec]) -> bytes:
    indexed = snapshot.set_index("sensor_key")
    rows = []
    for spec in specs:
        row = indexed.loc[spec.key]
        timestamp = pd.Timestamp(row["timestamp"])
        rows.append(
            {
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "sensor_key": spec.key,
                "pid_tag": spec.display_tag,
                "display_name": spec.display_name,
                "description": spec.description,
                "group_id": spec.group_id,
                "group_name": spec.group_name,
                "measurement": spec.measurement,
                "value": row["value"],
                "unit": spec.unit,
                "quality": row["quality"],
                "alarm_state": row["alarm_state"],
                "limit_LL": spec.limits.get("LL"),
                "limit_L": spec.limits.get("L"),
                "limit_H": spec.limits.get("H"),
                "limit_HH": spec.limits.get("HH"),
                "pid_x_pct": spec.pid_highlight.x,
                "pid_y_pct": spec.pid_highlight.y,
                "pid_shape": spec.pid_highlight.shape,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def render_pid_finder(specs: Sequence[SensorSpec]) -> None:
    """Compact fallback so every one of the 16 sensors can open the P&ID dialog."""
    specs_map = {spec.key: spec for spec in specs}

    def _label(key: str) -> str:
        spec = specs_map[key]
        return f"{spec.display_tag}  {spec.display_name}"

    with st.container(key="pid_finder_tool"):
        selector_col, button_col = st.columns(
            [0.82, 0.18], gap="small", vertical_alignment="bottom"
        )
        with selector_col:
            selected_key = st.selectbox(
                "Locate sensor on P&ID",
                options=[spec.key for spec in specs],
                format_func=_label,
                key="pid_sensor_finder",
                help="Choose any of the 16 dashboard sensors.",
            )
        with button_col:
            st.button(
                "View",
                key="pid_sensor_finder_button",
                help=f"Locate {specs_map[selected_key].display_tag} on P&ID",
                icon=":material/location_searching:",
                type="secondary",
                width="stretch",
                on_click=request_pid_dialog,
                args=(selected_key,),
            )


def render_all_sensors_tab(
    snapshot: pd.DataFrame, specs: Sequence[SensorSpec], scenario_name: str
) -> None:
    render_pid_finder(specs)
    table, states = build_sensor_table(snapshot, specs)
    st.html(
        _data_table_html(
            table,
            sensor_table_styles(table, states),
            aria_label="All sensors",
            max_height=575,
            min_width=980,
        )
    )
    stamp = pd.Timestamp.now(tz=DISPLAY_TZ).strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "Download snapshot",
        data=snapshot_csv(snapshot, specs),
        file_name=f"hydrostar_snapshot_{_slug(scenario_name)}_{stamp}.csv",
        mime="text/csv",
        key="download_snapshot",
        type="tertiary",
        icon=":material/download:",
        on_click="ignore",
    )


# ---------------------------------------------------------------------------
# 9. Trend charts and event log
# ---------------------------------------------------------------------------


def downsample(frame: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """Stride-based downsampling that keeps genuine samples (and their NaN
    gaps) instead of averaging across missing data."""
    if len(frame) <= max_points:
        return frame
    step = max(1, math.ceil(len(frame) / max_points))
    positions = list(range(0, len(frame), step))
    if positions[-1] != len(frame) - 1:
        positions.append(len(frame) - 1)
    return frame.iloc[positions].reset_index(drop=True)


def chart_theme(light: bool) -> dict[str, str]:
    if light:
        return {
            "panel": "#ffffff",
            "font": "#1a2010",
            "grid": "rgba(0,0,0,0.08)",
            "line": "rgba(0,0,0,0.20)",
            "hover_bg": "#f0f7e0",
            "hover_border": "rgba(73,152,35,0.45)",
        }
    return {
        "panel": "#171d22",
        "font": "#f2f4f7",
        "grid": "rgba(255,255,255,0.08)",
        "line": "rgba(255,255,255,0.18)",
        "hover_bg": "#111821",
        "hover_border": "rgba(167,215,48,0.35)",
    }


def build_trend_chart(
    spec: SensorSpec,
    sensor_frame: pd.DataFrame,
    window_label: str,
    scenario_name: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    light: bool,
) -> go.Figure:
    theme = chart_theme(light)
    axis_font = {"color": theme["font"], "family": FONT_STACK, "size": 11}
    decimals = spec.decimals

    x_values = sensor_frame["timestamp"].dt.tz_localize(None)
    y_values = sensor_frame["value"]
    custom = np.column_stack(
        (
            sensor_frame["quality"].astype(str),
            sensor_frame["alarm_state"].astype(str),
        )
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=spec.display_tag,
            line={
                "color": MEASUREMENT_LINE_COLOURS.get(spec.measurement, WATER_BLUE),
                "width": 2.2,
            },
            connectgaps=False,  # preserve missing-data gaps
            customdata=custom,
            hovertemplate=(
                "%{x|%d %b %H:%M:%S}<br>"
                + f"{spec.display_name}: %{{y:.{decimals}f}} {spec.unit}<br>"
                + "Alarm: %{customdata[1]}<br>"
                + "Quality: %{customdata[0]}<br>"
                + f"<extra>{spec.display_tag}</extra>"
            ),
        )
    )

    limits = spec.limits
    finite = y_values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    candidates = [value for value in limits.values() if value is not None]
    if finite.size:
        candidates.extend([float(finite.min()), float(finite.max())])
    if candidates:
        y_low = min(candidates)
        y_high = max(candidates)
        span = y_high - y_low
        pad = span * 0.14 if span > 0 else max(abs(y_high), 1.0) * 0.25
        y_bottom, y_top = y_low - pad, y_high + pad
        figure.update_yaxes(range=[y_bottom, y_top])
    else:
        y_bottom, y_top = 0.0, 1.0

    high = limits.get("H")
    high_high = limits.get("HH")
    low = limits.get("L")
    low_low = limits.get("LL")

    # Threshold lines with labels, corner positions chosen so the four level
    # labels (LL, L, H, HH) do not sit on top of each other.
    threshold_layout = (
        ("HH", high_high, RED, "dot", "top right"),
        ("H", high, AMBER, "dash", "top left"),
        ("L", low, AMBER, "dash", "bottom left"),
        ("LL", low_low, RED, "dot", "bottom right"),
    )
    for name, level, colour, dash, position in threshold_layout:
        if level is None:
            continue
        figure.add_hline(
            y=level,
            line={"color": colour, "width": 1.4, "dash": dash},
            annotation_text=f"{name} = {level:g} {spec.unit}",
            annotation_position=position,
            annotation_font={"color": colour, "size": 10, "family": FONT_STACK},
        )
    if not spec.configured:
        figure.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.98,
            text=NO_APPROVED_LIMITS, showarrow=False,
            font={"color": LIGHT_GREY, "size": 11, "family": FONT_STACK},
        )

    figure.update_layout(
        title={
            "text": f"{spec.display_tag} {spec.display_name} ({spec.unit})",
            "font": {"color": theme["font"], "size": 14, "family": FONT_STACK},
            "x": 0.01,
        },
        height=380,
        margin={"l": 54, "r": 38, "t": 44, "b": 44},
        paper_bgcolor=theme["panel"],
        plot_bgcolor=theme["panel"],
        font={"family": FONT_STACK, "color": theme["font"], "size": 12},
        showlegend=False,
        hoverlabel={
            "bgcolor": theme["hover_bg"],
            "bordercolor": theme["hover_border"],
            "font": {"color": theme["font"], "family": FONT_STACK, "size": 12},
        },
        # uirevision keeps zoom/pan across five-second refreshes; it resets
        # only when the sensor, window or scenario changes.
        uirevision=f"{spec.key}|{window_label}|{scenario_name}",
    )
    figure.update_xaxes(
        range=[
            window_start.tz_localize(None),
            window_end.tz_localize(None),
        ],
        gridcolor=theme["grid"],
        linecolor=theme["line"],
        zeroline=False,
        automargin=True,
        hoverformat="%d %b %Y at %H:%M:%S",
        tickfont=axis_font,
    )
    figure.update_yaxes(
        title_text=spec.unit,
        gridcolor=theme["grid"],
        linecolor=theme["line"],
        zeroline=False,
        automargin=True,
        tickfont=axis_font,
        title_font=axis_font,
    )
    return figure


def events_frame(events: list[dict[str, object]], specs_map: Mapping[str, SensorSpec]) -> pd.DataFrame:
    rows = []
    for event in reversed(events):  # newest first
        spec = specs_map.get(str(event["sensor_key"]))
        decimals = spec.decimals if spec else 2
        rows.append(
            {
                "Timestamp": format_timestamp(event["timestamp"], include_date=True),
                "P&ID tag": clean_ui_tag(event["pid_tag"]),
                "Sensor": event["sensor"],
                "Previous state": alarm_display_text(str(event["previous"])),
                "New state": alarm_display_text(str(event["new"])),
                "Value": format_number(event["value"], decimals),
                "Unit": event["unit"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Timestamp", "P&ID tag", "Sensor", "Previous state", "New state",
            "Value", "Unit",
        ],
    )


def event_table_styles(
    table: pd.DataFrame, events: list[dict[str, object]]
) -> pd.DataFrame:
    light_theme = is_light()
    base_background = "#ffffff" if light_theme else "#171d22"
    base_text = "#1a2010" if light_theme else "#f2f4f7"
    css = pd.DataFrame(
        f"background-color:{base_background}; color:{base_text};",
        index=table.index,
        columns=table.columns,
    )
    previous_col = table.columns.get_loc("Previous state")
    new_col = table.columns.get_loc("New state")
    ordered = list(reversed(events))
    for i, event in enumerate(ordered):
        for column, state in (
            (previous_col, str(event["previous"])),
            (new_col, str(event["new"])),
        ):
            background, foreground = state_badge_colours(state)
            css.iloc[i, column] = (
                f"background-color:{background}; color:{foreground}; font-weight:600;"
            )
    return css


def events_csv(events: list[dict[str, object]]) -> bytes:
    rows = [
        {
            "timestamp": pd.Timestamp(event["timestamp"]).isoformat(timespec="seconds"),
            "sensor_key": event["sensor_key"],
            "pid_tag": clean_ui_tag(event["pid_tag"]),
            "sensor": event["sensor"],
            "previous_state": event["previous"],
            "new_state": event["new"],
            "value": event["value"],
            "unit": event["unit"],
        }
        for event in reversed(events)
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def window_history_csv(window_frame: pd.DataFrame) -> bytes:
    export = window_frame.copy()
    export["timestamp"] = export["timestamp"].map(
        lambda value: pd.Timestamp(value).isoformat(timespec="seconds")
    )
    export = export.drop(columns=["source"], errors="ignore")
    if "pid_tag" in export:
        export["pid_tag"] = export["pid_tag"].map(clean_ui_tag)
    return export.to_csv(index=False).encode("utf-8")


def render_trends_tab(
    frame: pd.DataFrame,
    specs_map: Mapping[str, SensorSpec],
    selected: Sequence[str],
    window_label: str,
    scenario_name: str,
    events: list[dict[str, object]],
) -> None:
    newest = frame["timestamp"].max()
    minutes = WINDOW_OPTIONS[window_label]
    cutoff = newest - pd.Timedelta(minutes, unit="m")
    window_frame = frame.loc[frame["timestamp"] >= cutoff]

    if not selected:
        st.caption("Select one or two sensors in the sidebar to view trends.")
    else:
        selected = list(selected)[:2]
        columns = st.columns(len(selected), gap="medium") if len(selected) == 2 else [
            st.container()
        ]
        for container, key in zip(columns, selected):
            spec = specs_map[key]
            sensor_frame = downsample(
                window_frame.loc[window_frame["sensor_key"] == key]
                .sort_values("timestamp", kind="mergesort")
                .reset_index(drop=True)
            )
            figure = build_trend_chart(
                spec, sensor_frame, window_label, scenario_name,
                cutoff, newest, is_light(),
            )
            with container:
                st.plotly_chart(
                    figure,
                    config=CHART_CONFIG,
                    key=f"trend_{key}",
                    width="stretch",
                )

    stamp = pd.Timestamp.now(tz=DISPLAY_TZ).strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "Download history",
        data=window_history_csv(window_frame),
        file_name=(
            f"hydrostar_history_{_slug(scenario_name)}_"
            f"{_slug(window_label)}_{stamp}.csv"
        ),
        mime="text/csv",
        key="download_history",
        help=f"Download {window_label} of readings for all 16 sensors",
        type="tertiary",
        icon=":material/download:",
        on_click="ignore",
    )

    st.markdown(
        '<h2 class="section-title">Alarm and event log</h2>', unsafe_allow_html=True
    )
    if not events:
        st.caption("No alarm state changes have been recorded in this session.")
        return
    table = events_frame(events, specs_map)
    st.html(
        _data_table_html(
            table,
            event_table_styles(table, events),
            aria_label="Alarm and event log",
            max_height=430,
            min_width=980,
        )
    )
    st.markdown(
        f'<p class="table-note">Showing the {len(table)} most recent alarm '
        "state changes.</p>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download event log",
        data=events_csv(events),
        file_name=f"hydrostar_events_{_slug(scenario_name)}_{stamp}.csv",
        mime="text/csv",
        key="download_events",
        type="tertiary",
        icon=":material/download:",
        on_click="ignore",
    )


# ---------------------------------------------------------------------------
# 10. Sidebar
# ---------------------------------------------------------------------------


def _reset_session() -> None:
    st.session_state.pop("sim", None)


def render_sidebar(specs: Sequence[SensorSpec]) -> str:
    specs_map = {spec.key: spec for spec in specs}

    if st.session_state.get("status_select") not in SCENARIOS:
        st.session_state.pop("status_select", None)

    legacy_selection = [
        key
        for key in st.session_state.pop("trend_sensors", DEFAULT_TREND_SENSORS)
        if key in specs_map
    ]
    default_primary = (
        legacy_selection[0] if legacy_selection else DEFAULT_TREND_SENSORS[0]
    )
    default_comparison = (
        legacy_selection[1] if len(legacy_selection) > 1 else None
    )
    if st.session_state.get("trend_sensor_primary", default_primary) not in specs_map:
        st.session_state.pop("trend_sensor_primary", None)

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title-card">'
            '<p class="sidebar-kicker">HydroStar</p>'
            '<p class="sidebar-title">Electrolyser monitoring</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="sidebar-section-label">Controls</p>',
            unsafe_allow_html=True,
        )
        saved_theme = st.query_params.get("theme", st.session_state.get("theme", "dark"))
        light_mode = st.toggle(
            "Light mode", value=saved_theme == "light", key="light_mode_toggle"
        )
        theme = "light" if light_mode else "dark"
        st.session_state["theme"] = theme
        if st.query_params.get("theme") != theme:
            st.query_params["theme"] = theme

        st.selectbox(
            "Status",
            list(STATUS_ORDER),
            index=list(STATUS_ORDER).index(DEFAULT_STATUS),
            key="status_select",
        )
        st.button(
            "Reset readings",
            key="reset_readings",
            on_click=_reset_session,
            type="tertiary",
            icon=":material/restart_alt:",
            width="stretch",
        )
        if getattr(st, "fragment", None) is None:  # pragma: no cover
            st.button("Refresh now", key="manual_refresh", width="stretch")
            st.caption(
                "This Streamlit version does not support automatic fragment "
                "refresh. Use Refresh now to load new readings."
            )

        st.markdown(
            '<p class="sidebar-section-label sidebar-section-break">Trend charts</p>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("time_range") not in WINDOW_OPTIONS:
            st.session_state.pop("time_range", None)
        st.selectbox(
            "Time range",
            list(WINDOW_OPTIONS),
            index=list(WINDOW_OPTIONS).index(DEFAULT_WINDOW),
            key="time_range",
        )

        def trend_option_label(key: Optional[str]) -> str:
            if key is None:
                return "No comparison"
            spec = specs_map[key]
            return f"{spec.display_tag}  {spec.display_name}"

        primary_sensor = st.selectbox(
            "Primary sensor",
            options=list(EXPECTED_KEYS),
            index=list(EXPECTED_KEYS).index(default_primary),
            format_func=trend_option_label,
            key="trend_sensor_primary",
        )
        comparison_options: list[Optional[str]] = [None] + [
            key for key in EXPECTED_KEYS if key != primary_sensor
        ]
        if st.session_state.get("trend_sensor_comparison") not in comparison_options:
            st.session_state.pop("trend_sensor_comparison", None)
        comparison_default = (
            default_comparison if default_comparison in comparison_options else None
        )
        st.selectbox(
            "Comparison sensor",
            options=comparison_options,
            index=comparison_options.index(comparison_default),
            format_func=trend_option_label,
            key="trend_sensor_comparison",
        )

        st.markdown(
            '<p class="sidebar-section-label sidebar-section-break">Alarm guide</p>',
            unsafe_allow_html=True,
        )
        legend = (
            (SECONDARY, "Normal"),
            (AMBER, "High or low warning"),
            (RED, "Critical high or low"),
            (LIGHT_GREY, "Unavailable or limits pending"),
        )
        st.markdown(
            "".join(
                '<div class="legend-row">'
                f'<span class="legend-dot" style="background:{colour};"></span>'
                f'<span class="legend-text">{html.escape(text)}</span></div>'
                for colour, text in legend
            ),
            unsafe_allow_html=True,
        )

    return theme


# ---------------------------------------------------------------------------
# 11. Internal validation (Section 20 of the build brief)
# ---------------------------------------------------------------------------


def run_self_checks(specs: Sequence[SensorSpec]) -> None:
    """Assert the alarm engine and registry behave exactly as specified.

    Runs silently on startup; ordinary users never see successful assertions.
    """
    level = {"LL": -10.0, "L": -5.0, "H": 5.0, "HH": 10.0}
    for value, expected in (
        (-11.0, "LL"), (-10.0, "LL"), (-7.0, "L"), (-5.0, "L"), (0.0, "NORMAL"),
        (5.0, "H"), (7.0, "H"), (10.0, "HH"), (11.0, "HH"),
    ):
        got = classify_alarm(value, level, QUALITY_GOOD)
        assert got == expected, f"level {value} -> {got}, expected {expected}"

    temperature = {"LL": None, "L": None, "H": 50.0, "HH": 60.0}
    for value, expected in (
        (49.9, "NORMAL"), (50.0, "H"), (59.9, "H"), (60.0, "HH"),
    ):
        got = classify_alarm(value, temperature, QUALITY_GOOD)
        assert got == expected, f"temperature {value} -> {got}, expected {expected}"

    pressure = {"LL": None, "L": None, "H": 2.5, "HH": 4.0}
    for value, expected in (
        (2.49, "NORMAL"), (2.50, "H"), (3.99, "H"), (4.00, "HH"),
    ):
        got = classify_alarm(value, pressure, QUALITY_GOOD)
        assert got == expected, f"pressure {value} -> {got}, expected {expected}"

    gas = {"LL": None, "L": None, "H": 2.0, "HH": 3.0}
    for value, expected in (
        (1.99, "NORMAL"), (2.00, "H"), (2.99, "H"), (3.00, "HH"),
    ):
        got = classify_alarm(value, gas, QUALITY_GOOD)
        assert got == expected, f"gas {value} -> {got}, expected {expected}"

    assert classify_alarm(None, level, QUALITY_GOOD) == STATE_UNKNOWN
    assert classify_alarm(float("nan"), level, QUALITY_GOOD) == STATE_UNKNOWN
    assert classify_alarm(25.0, temperature, QUALITY_MISSING) == STATE_UNKNOWN
    assert classify_alarm(25.0, temperature, QUALITY_STALE) == STATE_UNKNOWN
    assert classify_alarm(25.0, temperature, QUALITY_BAD) == STATE_UNKNOWN
    unconfigured = {"LL": None, "L": None, "H": None, "HH": None}
    assert classify_alarm(1.8, unconfigured, QUALITY_GOOD) == STATE_UNCONFIGURED

    assert worst_state(["H", "HH", "NORMAL"]) == "HH"
    assert worst_state(["L", "LL"]) == "LL"
    assert worst_state(["NORMAL", "H"]) == "H"
    assert worst_state(["NORMAL", "UNKNOWN", "UNCONFIGURED"]) == "NORMAL"
    assert worst_state(["UNCONFIGURED", "UNKNOWN"]) == "UNCONFIGURED"
    assert worst_state([]) == STATE_UNKNOWN

    # Registry expectations: exactly 16 unique keys, and the duplicated
    # displayed tag FT-003 kept as two separate instruments.
    keys = [spec.key for spec in specs]
    assert len(keys) == 16, f"expected 16 sensors, found {len(keys)}"
    assert len(set(keys)) == 16, "internal sensor keys are not unique"
    assert all(spec.description.strip() for spec in specs), (
        "every sensor requires a non-empty P&ID description"
    )
    assert all(0.0 <= spec.pid_highlight.x <= 100.0 for spec in specs), (
        "every P&ID x coordinate must be a percentage between 0 and 100"
    )
    assert all(0.0 <= spec.pid_highlight.y <= 100.0 for spec in specs), (
        "every P&ID y coordinate must be a percentage between 0 and 100"
    )
    analyser_shapes = {
        spec.key: spec.pid_highlight.shape
        for spec in specs
        if spec.key in {"ANALYSER_H2_IN_O2", "ANALYSER_O2_IN_H2"}
    }
    assert set(analyser_shapes.values()) == {"rectangle"}, (
        "both gas analyser P&ID highlights must be rectangles"
    )
    assert all(
        spec.pid_highlight.shape == "circle"
        for spec in specs
        if spec.key not in {"ANALYSER_H2_IN_O2", "ANALYSER_O2_IN_H2"}
    ), "ordinary instrument P&ID highlights must remain circular"
    ft003 = {spec.key: spec.pid_tag for spec in specs if spec.pid_tag == "FT-003"}
    assert set(ft003) == {"FT003_O2", "FT003_N2"}, (
        "FT-003 must map to exactly FT003_O2 and FT003_N2"
    )

    # Event extraction: split processing equals single-pass processing, and a
    # block with no state change appends nothing.
    demo_times = pd.date_range("2026-01-01", periods=6, freq="5s", tz="UTC")
    demo_block = pd.DataFrame(
        {
            "timestamp": demo_times,
            "sensor_key": ["X"] * 6,
            "pid_tag": ["X-001"] * 6,
            "display_name": ["Demo sensor"] * 6,
            "value": [0.0, 0.0, 6.0, 6.0, 11.0, 11.0],
            "unit": ["mm"] * 6,
            "alarm_state": ["NORMAL", "NORMAL", "H", "H", "HH", "HH"],
        }
    )
    single_events: list[dict[str, object]] = []
    collect_events(demo_block, {}, single_events)
    split_events: list[dict[str, object]] = []
    split_state: dict[str, str] = {}
    collect_events(demo_block.iloc[:3], split_state, split_events)
    collect_events(demo_block.iloc[3:], split_state, split_events)
    assert len(single_events) == 2, f"expected 2 events, got {len(single_events)}"
    assert len(split_events) == len(single_events), "split processing duplicated events"
    unchanged: list[dict[str, object]] = []
    collect_events(demo_block.iloc[4:], {"X": "HH"}, unchanged)
    assert not unchanged, "unchanged states must not append events"


# ---------------------------------------------------------------------------
# 12. Live dashboard (five-second fragment refresh with fallback)
# ---------------------------------------------------------------------------


def live_fragment(func: Callable[..., None]) -> Callable[..., None]:
    fragment = getattr(st, "fragment", None)
    if fragment is None:  # pragma: no cover - older Streamlit fallback
        return func
    return fragment(run_every=REFRESH_INTERVAL)(func)


@live_fragment
def render_dashboard(specs: Sequence[SensorSpec]) -> None:
    specs_map = {spec.key: spec for spec in specs}
    status_name = str(st.session_state.get("status_select", DEFAULT_STATUS))
    window_label = str(st.session_state.get("time_range", DEFAULT_WINDOW))
    if window_label not in WINDOW_OPTIONS:
        window_label = DEFAULT_WINDOW
    selected = []
    for key in (
        st.session_state.get("trend_sensor_primary", DEFAULT_TREND_SENSORS[0]),
        st.session_state.get("trend_sensor_comparison", DEFAULT_TREND_SENSORS[1]),
    ):
        if key in specs_map and key not in selected:
            selected.append(key)
    ensure_sim(status_name, specs)
    advance_sim(specs)
    sim = st.session_state["sim"]
    frame: pd.DataFrame = sim["frame"]
    snapshot = latest_snapshot(frame, specs)

    render_header(snapshot, frame)
    render_stack_row(sim["elec"], snapshot)

    with st.container(key="monitoring_overview"):
        rail_column, cards_column = st.columns([0.25, 0.75], gap="medium")
        with rail_column:
            with st.container(key="monitoring_safety"):
                render_safety_rail(snapshot)
        with cards_column:
            with st.container(key="monitoring_sensors"):
                render_critical_cards(snapshot, specs_map)

    all_sensors_tab, trends_tab = st.tabs(["All sensors", "Trends and events"])
    with all_sensors_tab:
        render_all_sensors_tab(snapshot, specs, status_name)
    with trends_tab:
        render_trends_tab(
            frame, specs_map, selected, window_label, status_name, sim["events"]
        )

    requested_pid_key = st.session_state.get("pid_dialog_sensor")
    if requested_pid_key in specs_map:
        snapshot_indexed = snapshot.set_index("sensor_key")
        show_pid_dialog(
            specs_map[str(requested_pid_key)],
            snapshot_indexed.loc[str(requested_pid_key)],
        )


# ---------------------------------------------------------------------------
# 13. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=str(LOGO_FILE) if LOGO_FILE.exists() else None,
        layout="wide",
        initial_sidebar_state="auto",
    )
    try:
        specs = load_registry(REGISTRY_FILE)
    except RegistryError as exc:
        st.error(str(exc))
        st.stop()
        return
    try:
        run_self_checks(specs)
    except AssertionError as exc:
        st.error(
            "Internal validation failed, so the dashboard was stopped rather "
            f"than show incorrect alarm behaviour. Details: {exc}"
        )
        st.stop()
        return

    theme = render_sidebar(specs)
    inject_css(theme)
    render_dashboard(specs)


if __name__ == "__main__":
    main()
