#!/usr/bin/env python3
"""
Manual native voice message sender for Discord, closely following the reference flow:
- Convert input audio to Ogg Opus (48kHz, 64k) with ffmpeg
- Request upload URL via attachments.create
- Upload via signed upload_url (PUT, Content-Type: audio/ogg)
- Send message with flags=8192 and voice attachment metadata

Usage:
  uv run python utils/send_voice_memo.py --channel <CHANNEL_ID> --file <PATH_TO_AUDIO>

Requirements:
- ffmpeg and ffprobe installed and available on PATH
- Environment: DISCORD_TOKEN must be set (bot token)

Notes:
- Uses a static User-Agent (no extra dependencies)
- Authorization header includes 'Bot ' prefix (required by Discord)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

USER_AGENT = "DiscordVoiceMemoTest/1.0"
API_BASE = "https://discord.com/api/v10"


def ffmpeg_to_ogg(input_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-c:a",
        "libopus",
        "-b:a",
        "64k",
        "-ar",
        "48000",
        str(out_path),
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-hide_banner",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    dur = (
        float(data["format"]["duration"])
        if "format" in data and "duration" in data["format"]
        else 0.0
    )
    # Reference rounds to integer seconds
    return float(int(round(dur)))


def random_waveform_b64() -> str:
    # Reference uses 256 random bytes; Discord expects base64 string up to 256 bytes
    return base64.b64encode(os.urandom(256)).decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, required=True, help="Target channel ID")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to input audio (wav/mp3/etc)"
    )
    args = parser.parse_args()

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("DISCORD_TOKEN not set in environment", file=sys.stderr)
        return 2

    in_path = Path(args.file)
    if not in_path.exists():
        print(f"File not found: {in_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        ogg_path = Path(td) / "voice-message.ogg"
        try:
            ffmpeg_to_ogg(in_path, ogg_path)
            duration = ffprobe_duration_seconds(ogg_path)
            waveform = random_waveform_b64()
            file_size = ogg_path.stat().st_size
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg/ffprobe error: {e}", file=sys.stderr)
            return 3

        # Step 1: Request upload URL
        url = f"{API_BASE}/channels/{args.channel}/attachments"
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        payload = {
            "files": [
                {
                    "filename": "voice-message.ogg",
                    "file_size": file_size,
                    "id": 0,
                }
            ]
        }
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            print(
                f"Failed to get upload URL: {r.status_code} {r.text}", file=sys.stderr
            )
            return 4
        data = r.json()["attachments"][0]
        upload_url = data["upload_url"]
        uploaded_filename = data.get("upload_filename") or data.get("uploaded_filename")
        if not upload_url or not uploaded_filename:
            print(f"Invalid attachments.create response: {r.text}", file=sys.stderr)
            return 4

        # Step 2: Upload the file
        with open(ogg_path, "rb") as fh:
            up = requests.put(
                upload_url, headers={"Content-Type": "audio/ogg"}, data=fh
            )
        if up.status_code not in (200, 201):
            print(f"Failed to upload file: {up.status_code} {up.text}", file=sys.stderr)
            return 5

        # Step 3: Send the message with voice flag
        msg_url = f"{API_BASE}/channels/{args.channel}/messages"
        msg_headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        msg_payload = {
            "flags": 8192,
            "attachments": [
                {
                    "id": "0",
                    "filename": "voice-message.ogg",
                    "uploaded_filename": uploaded_filename,
                    "duration_secs": duration,
                    "waveform": waveform,
                }
            ],
        }
        mr = requests.post(msg_url, headers=msg_headers, json=msg_payload)
        if mr.status_code != 200:
            print(f"Error: {mr.status_code}\n{mr.text}", file=sys.stderr)
            return 6
        print("Upload Successful!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
