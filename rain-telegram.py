#!/usr/bin/env python3
"""
Rain Telegram Bot — message Rain from your phone via Telegram.

Setup:
  1. Copy .env.example to .env and fill in your values
  2. pip install python-telegram-bot httpx python-dotenv
  3. Make sure Rain's server is running: python3 server.py
  4. Run: python3 rain-telegram.py

Slash commands:
  /start       — greeting
  /status      — ping Rain's server
  /settings    — show current toggle state
  /sandbox     — toggle sandboxed code execution
  /websearch   — toggle live web search
  /verbose     — toggle verbose reflection output (prints to server terminal)
  /voicein     — toggle voice-to-text (send voice messages, Rain transcribes them)
  /voiceout    — toggle text-to-voice responses (requires Phase 8 TTS — not yet available)
  /help        — show this list
"""

import os
import json
import asyncio
import tempfile
import httpx
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CommandHandler,
    ContextTypes, filters,
)
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_ID = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))
RAIN_URL   = os.environ.get("RAIN_URL", "http://localhost:7734")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Per-chat toggle state — lives in memory for the bot process lifetime.
# Keys: sandbox, websearch, verbose, voice_in, voice_out
_DEFAULT_STATE = {
    "sandbox":   False,
    "websearch": False,
    "verbose":   False,
    "voice_in":  False,
    "voice_out": False,
}
_state: dict[int, dict] = {}


def _get_state(chat_id: int) -> dict:
    if chat_id not in _state:
        _state[chat_id] = dict(_DEFAULT_STATE)
    return _state[chat_id]


def _fmt_state(s: dict) -> str:
    def _icon(v): return "✅" if v else "⬜"
    return (
        f"{_icon(s['sandbox'])} /sandbox — code execution\n"
        f"{_icon(s['websearch'])} /websearch — live web search\n"
        f"{_icon(s['verbose'])} /verbose — verbose reflection (server terminal)\n"
        f"{_icon(s['voice_in'])} /voicein — voice → text input\n"
        f"{_icon(s['voice_out'])} /voiceout — text → voice response _(Phase 8, not yet available)_"
    )


def _check_config():
    if not BOT_TOKEN:
        raise SystemExit("❌ TELEGRAM_BOT_TOKEN not set in .env")
    if not ALLOWED_ID:
        raise SystemExit("❌ TELEGRAM_CHAT_ID not set in .env")


def _auth(update: Update) -> bool:
    return update.effective_chat.id == ALLOWED_ID


# ── Rain query ─────────────────────────────────────────────────────────────────

async def _query_rain(message: str, chat_id: int) -> str:
    """Send a message to Rain's /api/chat SSE endpoint with current toggle state."""
    s = _get_state(chat_id)
    payload = {
        "message":    message,
        "session_id": f"telegram_{chat_id}",
        "sandbox":    s["sandbox"],
        "web_search": s["websearch"],
        "verbose":    s["verbose"],
    }
    full_response = []
    error = None

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{RAIN_URL}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    t = data.get("type")
                    if t == "token":
                        full_response.append(data.get("content", ""))
                    elif t == "done":
                        final = data.get("response", "").strip()
                        if final:
                            return final
                    elif t == "error":
                        error = data.get("message", "Unknown error from Rain")

    except httpx.ConnectError:
        return "⚠️ Can't reach Rain's server. Is `python3 server.py` running?"
    except Exception as e:
        return f"⚠️ Error: {e}"

    if error:
        return f"⚠️ Rain error: {error}"
    result = "".join(full_response).strip()
    return result if result else "⚠️ Rain returned an empty response."


async def _transcribe_ogg(ogg_bytes: bytes) -> str | None:
    """POST an OGG voice file to Rain's /api/transcribe endpoint. Returns text or None."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            files = {"audio": ("voice.ogg", ogg_bytes, "audio/ogg")}
            r = await client.post(f"{RAIN_URL}/api/transcribe", files=files)
            if r.status_code == 200:
                data = r.json()
                return data.get("text", "").strip() or None
            elif r.status_code == 503:
                return None  # whisper not installed
    except Exception as e:
        log.warning(f"Transcription failed: {e}")
    return None


# ── Message handlers ────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        log.warning(f"Blocked message from unauthorized chat_id: {update.effective_chat.id}")
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    log.info(f"→ Rain: {user_text[:80]}")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = await _query_rain(user_text, update.effective_chat.id)
    log.info(f"← Rain: {response[:80]}")
    await _send_long(update, response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages when voice_in is enabled."""
    if not _auth(update):
        return

    s = _get_state(update.effective_chat.id)
    if not s["voice_in"]:
        await update.message.reply_text(
            "🎙️ Voice input is off. Enable it with /voicein"
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Download OGG from Telegram
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    ogg_bytes = bytes(await voice_file.download_as_bytearray())

    transcription = await _transcribe_ogg(ogg_bytes)
    if not transcription:
        await update.message.reply_text(
            "⚠️ Transcription failed. Make sure faster-whisper is installed:\n"
            "`.venv/bin/pip install faster-whisper`"
        )
        return

    log.info(f"🎙️ Transcribed: {transcription[:80]}")
    await update.message.reply_text(f"🎙️ _{transcription}_", parse_mode="Markdown")

    response = await _query_rain(transcription, update.effective_chat.id)
    await _send_long(update, response)


async def _send_long(update: Update, text: str):
    """Send a response, splitting at Telegram's 4096-char limit."""
    if len(text) <= 4096:
        await update.message.reply_text(text)
    else:
        for chunk in [text[i:i+4096] for i in range(0, len(text), 4096)]:
            await update.message.reply_text(chunk)


# ── Slash command handlers ──────────────────────────────────────────────────────

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update): return
    await update.message.reply_text(
        "⛈️ *Rain is online.*\n\n"
        "Send me a message — full pipeline, memory, tools.\n\n"
        "Type /help to see available commands.",
        parse_mode="Markdown"
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update): return
    await update.message.reply_text(
        "⛈️ *Rain Commands*\n\n"
        "/settings — show current toggle state\n"
        "/sandbox — toggle sandboxed code execution\n"
        "/websearch — toggle live web search\n"
        "/verbose — toggle verbose reflection (server terminal)\n"
        "/voicein — toggle voice-to-text input\n"
        "/voiceout — toggle voice responses _(Phase 8, not yet available)_\n"
        "/status — ping Rain's server\n"
        "/help — this message",
        parse_mode="Markdown"
    )


async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update): return
    s = _get_state(update.effective_chat.id)
    await update.message.reply_text(
        f"⚙️ *Current settings:*\n\n{_fmt_state(s)}",
        parse_mode="Markdown"
    )


async def _toggle(update: Update, key: str, label: str):
    """Generic toggle handler — flips a boolean state and confirms."""
    if not _auth(update): return
    s = _get_state(update.effective_chat.id)
    s[key] = not s[key]
    state_str = "ON ✅" if s[key] else "OFF ⬜"
    await update.message.reply_text(f"{label}: {state_str}")


async def handle_sandbox(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _toggle(update, "sandbox", "🧪 Sandbox")

async def handle_websearch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _toggle(update, "websearch", "🌐 Web search")

async def handle_verbose(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _toggle(update, "verbose", "📝 Verbose (server terminal)")

async def handle_voicein(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update): return
    # Check whether whisper is available before enabling
    s = _get_state(update.effective_chat.id)
    if not s["voice_in"]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{RAIN_URL}/api/voice-status")
                data = r.json()
                if not data.get("available"):
                    hint = data.get("install_hint", "")
                    await update.message.reply_text(
                        f"⚠️ Whisper not installed — voice input unavailable.\n"
                        f"To install: `{hint}`",
                        parse_mode="Markdown"
                    )
                    return
        except Exception:
            await update.message.reply_text("⚠️ Couldn't reach Rain's server to check voice status.")
            return
    await _toggle(update, "voice_in", "🎙️ Voice input")

async def handle_voiceout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update): return
    await update.message.reply_text(
        "🔇 Voice responses require Phase 8 (TTS via piper) — not yet implemented.\n"
        "It's next on the roadmap."
    )


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update): return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{RAIN_URL}/api/health")
            if r.status_code == 200:
                await update.message.reply_text("✅ Rain server is up.")
            else:
                await update.message.reply_text(f"⚠️ Rain server returned {r.status_code}.")
    except Exception:
        await update.message.reply_text("❌ Rain server is unreachable.")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    _check_config()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Slash commands
    app.add_handler(CommandHandler("start",     handle_start))
    app.add_handler(CommandHandler("help",      handle_help))
    app.add_handler(CommandHandler("status",    handle_status))
    app.add_handler(CommandHandler("settings",  handle_settings))
    app.add_handler(CommandHandler("sandbox",   handle_sandbox))
    app.add_handler(CommandHandler("websearch", handle_websearch))
    app.add_handler(CommandHandler("verbose",   handle_verbose))
    app.add_handler(CommandHandler("voicein",   handle_voicein))
    app.add_handler(CommandHandler("voiceout",  handle_voiceout))

    # Messages
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info(f"⛈️  Rain Telegram bot running (allowed chat_id: {ALLOWED_ID})")
    app.run_polling()


if __name__ == "__main__":
    main()
