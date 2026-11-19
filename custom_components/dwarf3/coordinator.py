"""Poll the relay for scope status."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import RelayBusy, RelayClient, RelayError, RelayUnauthorized, RelayUnreachable, ScopeError
from .const import CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL, DIRECT_DEFAULT_SCAN_INTERVAL, DOMAIN
from .direct import ScopeDirectClient, ScopeUnreachable
from .stack import StackLibrary, StackTracker

_LOGGER = logging.getLogger(__name__)

# The relay refuses a poll now and then (the DWARFLAB app dropping its connection mid-read, a Wi-Fi blip). Keep
# the last reading through that rather than turning every entity unavailable for one poll.
GRACE_S = 90

type Dwarf3ConfigEntry = ConfigEntry[Dwarf3Coordinator]

# The motor positions only come from a poll. The relay's push carries the last ones it remembers, which can be from
# the last time the scope was on, and taken as current they hide a scope that needs Prepare.
POLLED_ONLY = ("pitch", "rotation")


def _known(pushed: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """The pushed values worth keeping. A missing reading must not wipe one we have, but no job means no job."""
    return {k: v for k, v in pushed.items() if v is not None or k == "job" or k not in current}


class Dwarf3Coordinator(DataUpdateCoordinator[dict[str, Any]]):
    config_entry: Dwarf3ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: Dwarf3ConfigEntry, client: RelayClient) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=timedelta(seconds=entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)),
        )
        self.client = client
        self.relay_version: str | None = None
        self.platforms: list[Platform] = []  # what was set up, so a mode switch unloads the right ones
        self.stack: StackTracker | None = None
        self.library: StackLibrary | None = None
        self.sessions = Dwarf3SessionsCoordinator(hass, entry, client)
        self._last_ok = float("-inf")
        # The last status the relay pushed, and whether we are listening to it right now.
        self.live: dict[str, Any] = {}
        self.live_connected = False

    def apply_live(self, pushed: dict[str, Any]) -> None:
        """Take a status the relay pushed. The poll's schedule is left alone, so it still reads the motors."""
        pushed = {k: v for k, v in pushed.items() if k not in POLLED_ONLY}
        previous = self.data or {}
        self.live = pushed
        self._last_ok = time.monotonic()  # a push proves the relay is alive, whatever the next poll says
        reachable = pushed.get("scope_reachable")
        if reachable is not None and bool(reachable) != bool(previous.get("scope_reachable")):
            # The scope came or went. The positions we hold belong to the last time it was on, so wait for a poll.
            previous_kept = {k: v for k, v in previous.items() if k not in POLLED_ONLY}
        else:
            previous_kept = previous
        merged = {**previous_kept, **_known(pushed, previous_kept)}
        came_online = pushed.get("scope_reachable") and not previous.get("scope_reachable")
        if merged != previous or not self.last_update_success:
            self.data = merged
            self.last_update_success = True
            self.async_update_listeners()
        if came_online:
            # The relay sees the scope first. Read the motors and dark counts, which only a poll can do.
            self.hass.async_create_task(self.async_request_refresh())

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            polled = await self.client.status()
        except RelayUnauthorized as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except ScopeError:
            # The relay is up and the scope is not (asleep, or off the network, or still starting). Keep what the
            # relay told us, and otherwise just say the scope is away.
            return dict(self.live) if self.live_connected and self.live else {"scope_reachable": False}
        except (RelayBusy, RelayUnreachable) as err:
            if self.data and time.monotonic() - self._last_ok < GRACE_S:
                _LOGGER.debug("Keeping the last reading through a failed poll: %s", err)
                return self.data
            raise UpdateFailed(f"Relay did not answer: {err}") from err
        except RelayError as err:
            raise UpdateFailed(str(err)) from err
        self._last_ok = time.monotonic()
        # What was pushed is newer than what the poll read from the relay's memory a moment ago.
        return {**polled, **_known(self.live, polled)} if self.live_connected else polled


class Dwarf3SessionsCoordinator(DataUpdateCoordinator[list[str]]):
    """The capture sessions on the scope's SD card. Slow, because it reads the scope's album."""

    def __init__(self, hass: HomeAssistant, entry: Dwarf3ConfigEntry, client: RelayClient) -> None:
        super().__init__(
            hass, _LOGGER, config_entry=entry, name=f"{DOMAIN} sessions", update_interval=timedelta(minutes=10)
        )
        self.client = client

    async def _async_update_data(self) -> list[str]:
        try:
            return await self.client.sessions()
        except RelayUnauthorized as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except RelayError as err:
            raise UpdateFailed(str(err)) from err


SESSIONS_EVERY = timedelta(minutes=10)


class Dwarf3DirectCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Read-only status straight from the scope: reachability, SD card, and capture sessions."""

    config_entry: Dwarf3ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: Dwarf3ConfigEntry, client: ScopeDirectClient) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=timedelta(seconds=entry.options.get(CONF_SCAN_INTERVAL, DIRECT_DEFAULT_SCAN_INTERVAL)),
        )
        self.client = client
        self.platforms: list[Platform] = []
        self.stack: StackTracker | None = None
        self.library: StackLibrary | None = None
        self._sessions: list[str] | None = None
        self._sessions_at: float | None = None
        self._sessions_task: asyncio.Task[None] | None = None

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            data: dict[str, Any] = {"scope_reachable": True, **await self.client.sd_card()}
        except ScopeUnreachable:
            # Off, asleep, or out of range. Not a failure of ours, so the entities go unknown, not unavailable.
            return {"scope_reachable": False}
        now = self.hass.loop.time()
        stale = self._sessions_at is None or now - self._sessions_at > SESSIONS_EVERY.total_seconds()
        if stale and (self._sessions_task is None or self._sessions_task.done()):
            # Over FTP, and slow. A poll must not wait for it, so it fills in later.
            self._sessions_task = self.config_entry.async_create_background_task(
                self.hass, self._load_sessions(), "dwarf3 sessions"
            )
        data["sessions"] = self._sessions
        return data

    async def _load_sessions(self) -> None:
        try:
            self._sessions = await self.hass.async_add_executor_job(self.client.list_sessions_blocking)
        except OSError as err:
            _LOGGER.debug("Could not list capture sessions: %s", err)
            return
        self._sessions_at = self.hass.loop.time()
        self.async_set_updated_data({**(self.data or {}), "sessions": self._sessions})
