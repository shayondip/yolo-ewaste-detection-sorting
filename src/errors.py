"""Project-specific exceptions with user-friendly messages."""

from __future__ import annotations


class EwasteError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(EwasteError):
    """Configuration file is missing, unreadable or invalid."""


class DatasetError(EwasteError):
    """Dataset is missing or structurally invalid."""


class ModelNotFoundError(EwasteError):
    """Model weights could not be found or loaded."""


class MissingDependencyError(EwasteError):
    """An optional/required third-party package is not installed."""


class SourceError(EwasteError):
    """Input image/video/webcam source is invalid or unavailable."""


class UnknownClassError(EwasteError, KeyError):
    """A class name/ID is not defined in configs/config.yaml."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else "Unknown class"
