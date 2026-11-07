"""Reminder service, scheduler delivery, chat glue, tool and commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.reminders import scheduler as scheduler_mod
from bot.reminders.chat import handle_chat_message
from bot.reminders.scheduler import MAX_ATTEMPTS, PermanentDeliveryError, ReminderScheduler
from bot.reminders.service import Origin, ReminderService, Status
from bot.reminders.store import ReminderStore

NOW = datetime(2026, 11, 7, 18, 30, tzinfo=UTC)
USER = 1234
ORIGIN = Origin(user_id=USER, channel_id=55, guild_id=66, jump_url="https://discord.com/channels/66/55/77")


@pytest.fixture
def service(tmp_path):
    store = ReminderStore(tmp_path / "reminders.json")
    store.load()
    return ReminderService(store, max_per_user=3)


def _message(text: str = "", *, user_id: int = USER, reference=None):
    return SimpleNamespace(
        content=text,
        author=SimpleNamespace(id=user_id),
        channel=SimpleNamespace(id=55),
        guild=SimpleNamespace(id=66),
        jump_url="https://discord.com/channels/66/55/77",
        reference=reference,
        id=77,
    )


async def test_known_zone_schedules_and_persists(service, tmp_path):
    await service.set_timezone(USER, "America/New_York", now=NOW)
    outcome = await service.create(ORIGIN, "to post a fit pic on may 2027", now=NOW)
    assert outcome.status is Status.SCHEDULED
    assert outcome.reminder.due == datetime(2027, 5, 1, 13, 0, tzinfo=UTC)  # 09:00 EDT
    assert "<t:" in outcome.render() and "post a fit pic" in outcome.render()

    saved = json.loads((tmp_path / "reminders.json").read_text())
    assert saved["timezones"] == {str(USER): "America/New_York"}
    assert saved["reminders"][0]["task"] == "post a fit pic"

    reloaded = ReminderStore(tmp_path / "reminders.json")
    reloaded.load()
    assert reloaded.reminders[outcome.reminder.id].due == outcome.reminder.due


async def test_unknown_zone_waits_then_activates(service):
    outcome = await service.create(ORIGIN, "to post a fit pic on may 2027", now=NOW)
    assert outcome.status is Status.NEEDS_TIMEZONE
    assert "timezone" in outcome.render()

    activated, problems = await service.set_timezone(USER, "Europe/London", now=NOW + timedelta(minutes=5))
    assert problems == []
    assert [r.id for r in activated] == [outcome.reminder.id]
    assert activated[0].due == datetime(2027, 5, 1, 8, 0, tzinfo=UTC)  # 09:00 BST


async def test_waiting_reminder_whose_time_passed_fires_asap(service):
    outcome = await service.create(ORIGIN, "at 7pm call", now=NOW)
    later = NOW + timedelta(hours=3)  # user answered after 7pm London
    activated, _ = await service.set_timezone(USER, "Europe/London", now=later)
    assert activated[0].due == later


async def test_floating_time_needs_no_zone(service):
    outcome = await service.create(ORIGIN, "in 2 hours to stretch", now=NOW)
    assert outcome.status is Status.SCHEDULED
    assert outcome.reminder.due == NOW + timedelta(hours=2)


async def test_no_time_error_and_limit(service):
    assert (await service.create(ORIGIN, "to do stuff", now=NOW)).status is Status.NO_TIME
    assert (await service.create(ORIGIN, "on 31/02", now=NOW)).status is Status.FAILED
    for _ in range(3):
        await service.create(ORIGIN, "in 1 hour x", now=NOW)
    assert (await service.create(ORIGIN, "in 1 hour y", now=NOW)).status is Status.LIMIT


async def test_cancel_only_own(service):
    outcome = await service.create(ORIGIN, "in 1 hour x", now=NOW)
    assert await service.cancel(999, outcome.reminder.id) is None
    assert (await service.cancel(USER, outcome.reminder.id)).id == outcome.reminder.id
    assert service.list_for(USER) == []


async def test_reply_about_this_links_the_replied_message(service):
    origin = Origin(user_id=USER, channel_id=55, jump_url="https://x/own", referenced_url="https://x/ref")
    outcome = await service.create(origin, "about this tomorrow", now=NOW)
    assert outcome.reminder.task == "that message"
    assert outcome.reminder.jump_url == "https://x/ref"


async def test_scheduler_fires_due_and_keeps_future(service):
    due = (await service.create(ORIGIN, "in 1 minute now", now=NOW)).reminder
    future = (await service.create(ORIGIN, "in 3 hours later", now=NOW)).reminder
    sender = AsyncMock()
    sched = ReminderScheduler(MagicMock(), service, sender=sender)
    assert await sched.fire_due(NOW + timedelta(minutes=2)) == 1
    sender.assert_awaited_once()
    assert set(service.store.reminders) == {future.id}
    assert due.id not in service.store.reminders


async def test_scheduler_retries_transient_then_drops(service):
    reminder = (await service.create(ORIGIN, "in 1 minute x", now=NOW)).reminder
    error = discord.HTTPException(MagicMock(status=500, reason="boom"), "boom")
    sched = ReminderScheduler(MagicMock(), service, sender=AsyncMock(side_effect=error))
    moment = NOW + timedelta(minutes=2)
    await sched.fire_due(moment)
    assert reminder.attempts == 1 and reminder.due > moment
    for _ in range(MAX_ATTEMPTS):
        await sched.fire_due(reminder.due or moment)
    assert reminder.id not in service.store.reminders


async def test_scheduler_drops_permanent_failure(service):
    reminder = (await service.create(ORIGIN, "in 1 minute x", now=NOW)).reminder
    sched = ReminderScheduler(MagicMock(), service, sender=AsyncMock(side_effect=PermanentDeliveryError("dms closed")))
    await sched.fire_due(NOW + timedelta(minutes=2))
    assert reminder.id not in service.store.reminders


async def test_deliver_pings_only_owner_and_falls_back_to_dm(service, monkeypatch):
    reminder = (await service.create(ORIGIN, "in 1 minute @everyone party", now=NOW)).reminder
    sent = []

    async def fake_send(dest, content, **kwargs):
        sent.append((dest, content, kwargs["allowed_mentions"]))
        if dest == "channel":
            raise discord.Forbidden(MagicMock(status=403, reason="no"), "no")

    monkeypatch.setattr(scheduler_mod, "safe_send", fake_send)
    bot = MagicMock()
    bot.get_channel.return_value = "channel"
    bot.get_user.return_value = "user"
    await scheduler_mod.deliver(bot, reminder, NOW + timedelta(minutes=10))
    assert [d for d, _, _ in sent] == ["channel", "user"]
    _, content, mentions = sent[-1]
    assert f"<@{USER}>" in content and "late" in content
    assert mentions.everyone is False and mentions.roles is False
    assert [u.id for u in mentions.users] == [USER]


async def test_chat_flow_reminder_then_bare_city_reply(service):
    reply = await handle_chat_message(service, _message(), "remind me to post a fit pic on may 2027")
    assert "timezone" in reply
    reply = await handle_chat_message(service, _message(), "london")
    assert "Europe/London" in reply and "post a fit pic" in reply
    assert service.list_for(USER)[0].due == datetime(2027, 5, 1, 8, 0, tzinfo=UTC)


async def test_chat_ignores_non_reminders(service):
    assert await handle_chat_message(service, _message(), "remind me what we said") is None
    assert await handle_chat_message(service, _message(), "remind me about the plot") is None
    assert await handle_chat_message(service, _message(), "london") is None  # nothing waiting on a zone
    assert "when should I remind you" in await handle_chat_message(service, _message(), "remind me to do laundry")


async def test_chat_explicit_unknown_zone_explains(service):
    reply = await handle_chat_message(service, _message(), "my timezone is narnia")
    assert "don't recognise" in reply
    reply = await handle_chat_message(service, _message(), "my timezone is usa")
    assert "more than one timezone" in reply


async def test_set_reminder_tool(service):
    from bot.tools.builtins.reminders import set_reminder
    from bot.tools.types import ToolContext

    await service.set_timezone(USER, "Europe/London", now=NOW)
    ctx = ToolContext(message=_message(), bot=SimpleNamespace(reminder_service=service))
    result = await set_reminder(ctx, {"when": "friday 5pm", "what": "pay rent"})
    assert result.ok and "<t:" in result.content
    assert service.list_for(USER)[0].task == "pay rent"

    bad = await set_reminder(ctx, {"when": "whenever", "what": "x"})
    assert not bad.ok and "clearer time" in bad.error
    missing = await set_reminder(ToolContext(message=_message(), bot=SimpleNamespace()), {"when": "in 1 hour", "what": "x"})
    assert not missing.ok


async def test_commands_cog(service, monkeypatch):
    from bot.commands import reminder_commands as mod

    replies = []

    async def fake_reply(message, content, **kwargs):
        replies.append(content)

    monkeypatch.setattr(mod, "safe_reply", fake_reply)
    bot = SimpleNamespace()
    cog = mod.ReminderCommands(bot, service=service)
    assert bot.reminder_service is service
    ctx = SimpleNamespace(message=_message("!remindme in 2 hours stretch"), author=SimpleNamespace(id=USER))

    await mod.ReminderCommands.timezone.callback(cog, ctx, zone="new york")
    assert "America/New_York" in replies[-1]
    await mod.ReminderCommands.remindme.callback(cog, ctx, text="to post a fit pic on may 2027")
    assert "post a fit pic" in replies[-1] and "<t:" in replies[-1]
    await mod.ReminderCommands.reminders.callback(cog, ctx)
    assert "#1" in replies[-1]
    await mod.ReminderCommands.unremind.callback(cog, ctx, target="1")
    assert "cancelled reminder #1" in replies[-1]
    await mod.ReminderCommands.reminders_cancel.callback(cog, ctx, target="all")
    assert "cancelled 0" in replies[-1]


async def test_impossible_or_past_dates_fail_before_asking_for_zone(service):
    assert (await service.create(ORIGIN, "on 25/12/2020 xmas", now=NOW)).status is Status.FAILED
    assert (await service.create(ORIGIN, "today feed cat", now=NOW)).status is Status.FAILED


async def test_router_hook_only_answers_when_addressed(service):
    from bot.router import Router

    fake = SimpleNamespace(
        bot=SimpleNamespace(reminder_service=service, user=SimpleNamespace(id=42)),
        _is_mentioned=lambda m: m.mentioned,
        _is_reply_to_bot=lambda m: False,
        logger=MagicMock(),
    )
    msg = _message("remind me in 2 hours to stretch")
    msg.mentioned = False
    assert await Router._maybe_reminder(fake, msg, msg.content) is None
    msg.mentioned = True
    action = await Router._maybe_reminder(fake, msg, "<@42> remind me in 2 hours to stretch")
    assert action is not None and "stretch" in action.content
    assert len(service.list_for(USER)) == 1


def test_cog_commands_missing_from_command_map_are_delegated():
    """'!remindme' in a DM must reach the cog, not the LLM."""
    from bot.router import Router

    registered = {"remindme": object()}
    fake = SimpleNamespace(bot=SimpleNamespace(get_command=registered.get))
    assert Router._is_cog_command(fake, "!remindme to post a fit pic on may 2027")
    assert not Router._is_cog_command(fake, "!nosuchcommand hi")
    assert not Router._is_cog_command(fake, "<@42> !remindme x")  # prefix must lead
    assert not Router._is_cog_command(fake, "remind me in 2 hours")
    assert not Router._is_cog_command(SimpleNamespace(bot=MagicMock()), "!remindme x")  # test doubles
