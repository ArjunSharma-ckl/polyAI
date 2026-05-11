from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import provider_env_key
from .utils.env import load_env_file

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3


@dataclass
class DefaultModel:
    """Default provider and model pair used when omitted from AI(...)."""

    provider: Optional[str] = None
    model: Optional[str] = None


@dataclass
class ConfigStore:
    """Thread-safe global configuration store for API keys and defaults."""

    _keys: dict[str, str] = field(default_factory=dict)
    _default: DefaultModel = field(default_factory=DefaultModel)
    _env_loaded: bool = False
    _loaded_env_path: Optional[Path] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def configure(self, keys: Optional[dict[str, str]] = None, **kwargs: str) -> None:
        """Store provider API keys globally."""

        with self._lock:
            for provider, key in {**(keys or {}), **kwargs}.items():
                if key:
                    self._keys[provider.lower()] = key

    def get_key(self, provider: str, explicit: Optional[str] = None) -> Optional[str]:
        """Return an explicit, configured, or environment API key for ``provider``."""

        if explicit:
            return explicit
        self.ensure_env_loaded()
        with self._lock:
            configured = self._keys.get(provider.lower())
        if configured:
            return configured
        primary = provider_env_key(provider)
        alternatives = [
            primary,
            f"POLYAI_{provider.upper()}_KEY",
            f"POLYAI_{provider.upper()}_API_KEY",
        ]
        for name in alternatives:
            value = os.getenv(name)
            if value:
                return value
        return None

    def set_default(self, *, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Set default provider and model for future calls."""

        with self._lock:
            if provider is not None:
                self._default.provider = provider
            if model is not None:
                self._default.model = model

    def get_default(self) -> DefaultModel:
        """Return a copy of the current default provider/model pair."""

        with self._lock:
            return DefaultModel(provider=self._default.provider, model=self._default.model)

    def load_env(
        self, path: Optional[str | Path] = None, *, override: bool = False
    ) -> Optional[Path]:
        """Load API keys from a ``.env`` file."""

        loaded = load_env_file(path, override=override)
        with self._lock:
            self._env_loaded = True
            self._loaded_env_path = loaded
        return loaded

    def ensure_env_loaded(self) -> None:
        """Auto-load the nearest ``.env`` once."""

        with self._lock:
            already_loaded = self._env_loaded
        if already_loaded:
            return
        self.load_env()

    @property
    def loaded_env_path(self) -> Optional[Path]:
        """Return the path of the last loaded ``.env`` file."""

        with self._lock:
            return self._loaded_env_path


CONFIG = ConfigStore()


def configure(keys: Optional[dict[str, str]] = None, **kwargs: str) -> None:
    """Configure API keys globally."""

    CONFIG.configure(keys, **kwargs)


def get_api_key(provider: str, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve an API key for a provider."""

    return CONFIG.get_key(provider, explicit)


def set_default(*, provider: Optional[str] = None, model: Optional[str] = None) -> None:
    """Set default provider/model values."""

    CONFIG.set_default(provider=provider, model=model)


def get_default() -> DefaultModel:
    """Return configured defaults."""

    return CONFIG.get_default()


def load_env(path: Optional[str | Path] = None, *, override: bool = False) -> Optional[Path]:
    """Load a ``.env`` file explicitly."""

    return CONFIG.load_env(path, override=override)
