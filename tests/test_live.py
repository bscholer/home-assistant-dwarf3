"""The relay's live stream: parsed, applied at once, and quiet about reconnecting."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.dwarf3 import live
from custom_components.dwarf3.api import EventsUnsupported, RelayClient, RelayUnreachable
from custom_components.dwarf3.live import follow_relay

from .test_integration import BASE, STATUS, mock_relay, setup


async def test_events_are_parsed(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    body = (
        'event: status\ndata: {"battery_percent":70}\n\n'
        ": keepalive\n\n"
        'event: status\ndata: {"battery_percent":69,"job":null}\n\n'
        'event: other\ndata: {"ignored":true}\n\n'
        "event: status\ndata: not json\n\n"
    )
    aioclient_mock.get(f"{BASE}/api/events", text=body)
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    client = RelayClient(async_get_clientsession(hass), "relay.test", 9902)
    got = [event async for event in client.events()]
    assert got == [{"battery_percent": 70}, {"battery_percent": 69, "job": None}]


async def test_an_old_relay_has_no_stream(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    aioclient_mock.get(f"{BASE}/api/events", status=404, json={"detail": "Not Found"})
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    client = RelayClient(async_get_clientsession(hass), "relay.test", 9902)
    with pytest.raises(EventsUnsupported):
        async for _ in client.events():
            pass


async def test_a_push_shows_at_once_and_the_poll_keeps_it(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    mock_relay(aioclient_mock)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    assert hass.states.get("sensor.dwarf_3_battery").state == "87"

    coordinator.live_connected = True
    coordinator.apply_live({"battery_percent": 64, "tracking_state": 0, "job": None})
    assert hass.states.get("sensor.dwarf_3_battery").state == "64"
    assert hass.states.get("sensor.dwarf_3_capture_state") is not None
    assert coordinator.data["pitch"] == 172.5  # what only a poll can read is kept

    # The poll read the relay's memory a moment before the push, so the push wins.
    await coordinator.async_refresh()
    assert hass.states.get("sensor.dwarf_3_battery").state == "64"
    assert coordinator.data["job"] is None


PREPARED = {**STATUS, "job": None, "capture_state": 0, "pitch": 172.5, "rotation": 10.0}
UNPREPARED = {**PREPARED, "pitch": None, "rotation": None}


async def test_positions_from_the_last_session_do_not_hide_needs_prepare(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """The relay keeps the last motor positions it read and pushes them even after the scope slept. A scope that
    wakes up unprepared must still ask for Prepare."""
    mock_relay(aioclient_mock, status=PREPARED)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    coordinator.live_connected = True
    assert hass.states.get("binary_sensor.dwarf_3_needs_prepare").state == "off"

    old_positions = {"pitch": 172.5, "rotation": 10.0}
    coordinator.apply_live({"scope_reachable": False, **old_positions})
    assert hass.states.get("binary_sensor.dwarf_3_needs_prepare").state == "unknown"

    aioclient_mock.clear_requests()
    mock_relay(aioclient_mock, status=UNPREPARED)
    coordinator.apply_live({"scope_reachable": True, **old_positions})
    assert hass.states.get("binary_sensor.dwarf_3_needs_prepare").state != "off", "the old positions are not news"
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert hass.states.get("binary_sensor.dwarf_3_needs_prepare").state == "on"
    assert hass.states.get("sensor.dwarf_3_pitch").state == "unknown"


async def test_a_push_never_replaces_the_motor_positions_a_poll_read(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    mock_relay(aioclient_mock, status=PREPARED)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    coordinator.live_connected = True
    coordinator.apply_live({"pitch": 99.0, "rotation": 99.0, "battery_percent": 50})
    assert (coordinator.data["pitch"], coordinator.data["rotation"]) == (172.5, 10.0)
    await coordinator.async_refresh()
    assert (coordinator.data["pitch"], coordinator.data["rotation"]) == (172.5, 10.0)
    assert coordinator.data["battery_percent"] == 50


async def test_a_missing_reading_never_wipes_a_known_one(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    mock_relay(aioclient_mock)
    entry = await setup(hass)
    entry.runtime_data.apply_live({"battery_percent": None, "sd_free_gb": None})
    assert hass.states.get("sensor.dwarf_3_battery").state == "87"


async def test_a_push_does_not_move_the_poll_timer(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    """Pushes can come every second. If each reset the poll, the motors would never be read."""
    mock_relay(aioclient_mock)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    before = coordinator._unsub_refresh
    coordinator.apply_live({"battery_percent": 50})
    assert coordinator._unsub_refresh is before


async def test_a_scope_that_comes_online_triggers_a_read(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    mock_relay(aioclient_mock, status={"scope_reachable": False}, status_code=200)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    mock_calls = len([c for c in aioclient_mock.mock_calls if str(c[1]).endswith("/api/status")])
    coordinator.live_connected = True
    coordinator.apply_live({"scope_reachable": True, "battery_percent": 90})
    await hass.async_block_till_done()
    await asyncio.sleep(0)
    status_calls = len([c for c in aioclient_mock.mock_calls if str(c[1]).endswith("/api/status")])
    assert status_calls >= mock_calls  # the refresh is debounced, so it may land just after
    assert hass.states.get("sensor.dwarf_3_battery").state == "90"


async def test_a_failed_poll_keeps_what_the_relay_pushed(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker):
    mock_relay(aioclient_mock)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    coordinator.live_connected = True
    coordinator.live = {"scope_reachable": True, "battery_percent": 77}
    aioclient_mock.clear_requests()
    mock_relay(aioclient_mock, status={"detail": "scope did not answer"}, status_code=502)
    await coordinator.async_refresh()
    assert coordinator.data == {"scope_reachable": True, "battery_percent": 77}


def _client_with(*streams):
    """A fake client whose connections play the given scripts: lists of events, then an error or the end."""
    scripts = iter(streams)

    class Fake:
        async def events(self):
            script = next(scripts)
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item

    return Fake()


class FakeCoordinator:
    def __init__(self, client, hass):
        self.client, self.hass = client, hass
        self.live_connected = False
        self.applied: list[dict] = []
        self.refreshes = 0

    def apply_live(self, pushed):
        self.applied.append(pushed)

    async def async_request_refresh(self):
        self.refreshes += 1


async def test_reconnecting_is_quiet(hass: HomeAssistant, monkeypatch, caplog):
    sleeps: list[float] = []

    class Stop(Exception):
        pass

    async def fast_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 6:
            raise Stop

    monkeypatch.setattr(live, "asyncio", SimpleNamespace(sleep=fast_sleep, CancelledError=asyncio.CancelledError))
    down = RelayUnreachable("refused")
    client = _client_with(
        [{"n": 1}, down],  # lost
        [down],  # still down, still quiet
        [down],
        [{"n": 2}, down],  # back, then lost again
        [down],
        [{"n": 3}],  # back; ends normally, and the relay restarting is not an error either
        [down],
    )
    coordinator = FakeCoordinator(client, hass)
    with caplog.at_level(logging.DEBUG, logger="custom_components.dwarf3.live"):
        with pytest.raises(Stop):
            await follow_relay(coordinator)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert [a["n"] for a in coordinator.applied] == [1, 2, 3]
    assert len(warnings) == 3  # one per loss, never one per retry
    assert len(infos) == 2  # one per recovery
    assert sleeps[0] == 1.0 and max(sleeps) <= 15.0
    await hass.async_block_till_done()
    assert coordinator.refreshes == 2  # a read after each reconnect, not after the first connect


async def test_an_old_relay_ends_the_listener_quietly(hass: HomeAssistant, caplog):
    coordinator = FakeCoordinator(_client_with([EventsUnsupported("no")]), hass)
    with caplog.at_level(logging.INFO, logger="custom_components.dwarf3.live"):
        await follow_relay(coordinator)
    assert not caplog.records


async def test_a_push_counts_as_proof_of_life_for_a_refused_poll(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, monkeypatch
):
    """The relay sometimes refuses a poll with 409. While it keeps pushing, that is not an outage."""
    from custom_components.dwarf3 import coordinator as coordinator_module

    mock_relay(aioclient_mock)
    entry = await setup(hass)
    coordinator = entry.runtime_data
    monkeypatch.setattr(coordinator_module, "GRACE_S", 1000)
    coordinator._last_ok = float("-inf")  # the last good poll was long ago
    coordinator.live_connected = True
    coordinator.apply_live({"battery_percent": 61})

    aioclient_mock.clear_requests()
    mock_relay(aioclient_mock, status={"detail": "no app connection", "code": "busy"}, status_code=409)
    await coordinator.async_refresh()
    assert hass.states.get("sensor.dwarf_3_battery").state == "61"
