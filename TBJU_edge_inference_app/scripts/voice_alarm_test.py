#!/usr/bin/env python3
"""CLI test for USB speaker / voice alarm playback."""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alarm.voice_alarm import VoiceAlarmManager


def main():
    parser = argparse.ArgumentParser(description='Test voice alarm playback')
    parser.add_argument('--event', default='debris', help='alarm event key, e.g. debris/temp_high/test')
    parser.add_argument('--text', default='警告，发现异物', help='text to speak when no audio file is found')
    parser.add_argument('--audio-dir', default=str(PROJECT_ROOT / 'assets' / 'audio'))
    parser.add_argument('--wait', type=float, default=3.0, help='seconds to wait before exit')
    args = parser.parse_args()

    manager = VoiceAlarmManager(Path(args.audio_dir), enabled=True)
    queued = manager.trigger(args.event, args.text, cooldown_s=0, force=True)
    print(f'queued={queued} event={args.event} text={args.text}')
    print(f'audio_dir={args.audio_dir}')
    time.sleep(max(args.wait, 0.5))
    manager.close()


if __name__ == '__main__':
    main()

