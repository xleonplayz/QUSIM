# -*- coding: utf-8 -*-
"""
Compatibility layer - replaces Qudi dependencies for standalone operation.

This module provides local implementations of Qudi classes and utilities
so the mock server can run without requiring the full Qudi framework.
"""

from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional
import os
import re
import datetime
import yaml


# =============================================================================
# From qudi.interface.simple_laser_interface
# =============================================================================

class LaserState(IntEnum):
    """Laser operational states."""
    OFF = 0
    ON = 1
    LOCKED = 2
    UNKNOWN = -1


class ShutterState(IntEnum):
    """Laser shutter states."""
    CLOSED = 0
    OPEN = 1
    NO_SHUTTER = -1


class ControlMode(IntEnum):
    """Laser control modes."""
    POWER = 0
    CURRENT = 1
    MIXED = 2


# =============================================================================
# From qudi.interface.pulser_interface
# =============================================================================

class SequenceOption(Enum):
    """Sequence option types for pulser."""
    OPTIONAL = 'optional'
    MANDATORY = 'mandatory'
    FORCED = 'forced'
    NON_OPTIONAL = 'non_optional'


@dataclass
class ScalarConstraint:
    """Constraint for a scalar value with min, max, step, and default."""
    min: float = 0.0
    max: float = 0.0
    step: float = 0.0
    default: float = 0.0


@dataclass
class PulserConstraints:
    """
    Container for pulser hardware constraints.

    Replaces qudi.interface.pulser_interface.PulserConstraints
    """
    sample_rate: ScalarConstraint = field(default_factory=ScalarConstraint)
    a_ch_amplitude: ScalarConstraint = field(default_factory=ScalarConstraint)
    a_ch_offset: ScalarConstraint = field(default_factory=ScalarConstraint)
    d_ch_low: ScalarConstraint = field(default_factory=ScalarConstraint)
    d_ch_high: ScalarConstraint = field(default_factory=ScalarConstraint)
    waveform_length: ScalarConstraint = field(default_factory=ScalarConstraint)
    waveform_num: ScalarConstraint = field(default_factory=ScalarConstraint)
    sequence_num: ScalarConstraint = field(default_factory=ScalarConstraint)
    subsequence_num: ScalarConstraint = field(default_factory=ScalarConstraint)
    sequence_steps: ScalarConstraint = field(default_factory=ScalarConstraint)
    repetitions: ScalarConstraint = field(default_factory=ScalarConstraint)

    event_triggers: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    activation_config: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    sequence_option: SequenceOption = SequenceOption.OPTIONAL


# =============================================================================
# From qudi.util.datastorage
# =============================================================================

def get_timestamp_filename(timestamp: Optional[datetime.datetime] = None,
                           fmt: str = '%Y%m%d-%H%M-%S') -> str:
    """
    Generate a filename based on timestamp.

    Replaces qudi.util.datastorage.get_timestamp_filename
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime(fmt)


def create_dir_for_file(file_path: str) -> None:
    """
    Create directory structure for a file path if it doesn't exist.

    Replaces qudi.util.datastorage.create_dir_for_file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# =============================================================================
# From qudi.util.helpers
# =============================================================================

def natural_sort(iterable):
    """
    Sort strings in natural order (e.g., a1, a2, a10 instead of a1, a10, a2).

    Replaces qudi.util.helpers.natural_sort
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', str(text))]

    return sorted(iterable, key=natural_keys)


# =============================================================================
# From qudi.util.yaml
# =============================================================================

def yaml_dump(file_path: str, data, **kwargs) -> None:
    """
    Dump data to a YAML file.

    Replaces qudi.util.yaml.yaml_dump
    """
    create_dir_for_file(file_path)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, **kwargs)


def yaml_load(file_path: str):
    """
    Load data from a YAML file.

    Replaces qudi.util.yaml.yaml_load
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
