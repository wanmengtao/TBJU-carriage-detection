#!/usr/bin/env python3
"""LD3320 UART voice module test utility."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voice.ld3320_uart import (  # noqa: E402
    LD3320SerialReader,
    VoiceCommandMapper,
    decode_payload,
    describe_serial_frame,
)


def parse_args():
    ap = argparse.ArgumentParser(description='Test LD3320 UART voice commands')
    ap.add_argument('--port', default='/dev/ttyS9', help='serial port, e.g. /dev/ttyS9 or COM3')
    ap.add_argument('--baudrate', type=int, default=9600, help='serial baudrate')
    ap.add_argument('--config', default=str(ROOT / 'config' / 'voice_commands.json'))
    ap.add_argument('--simulate', help='simulate a text payload, e.g. 开始检测')
    ap.add_argument('--simulate_hex', help='simulate hex payload, e.g. 01 or aabb01')
    return ap.parse_args()


def show_match(mapper, data):
    text = decode_payload(data)
    hx = data.hex(' ')
    print(f'RAW text={text!r} hex={hx}')
    frame = describe_serial_frame(data)
    if frame:
        details = [f'type={frame.kind}']
        if frame.length is not None:
            details.append(f'len={frame.length}')
        if frame.command is not None:
            details.append(f'cmd=0x{frame.command:02x}')
        if frame.encoding is not None:
            details.append(f'enc=0x{frame.encoding:02x}')
        if frame.text:
            details.append(f'text={frame.text!r}')
        print('FRAME ' + ' '.join(details))
    match = mapper.match(data)
    if match:
        print(f'MATCH action={match.action} label={match.label} rule={match.matched_rule}')
    else:
        print('NO MATCH')


def main():
    args = parse_args()
    mapper = VoiceCommandMapper.from_file(Path(args.config))

    if args.simulate is not None:
        show_match(mapper, args.simulate.encode('utf-8'))
        return

    if args.simulate_hex is not None:
        compact = args.simulate_hex.replace(' ', '').replace('0x', '').replace(':', '')
        show_match(mapper, bytes.fromhex(compact))
        return

    print('Configured commands:')
    for line in mapper.describe_commands():
        print('  ' + line)
    print(f'\nOpening {args.port} @ {args.baudrate} ...')
    reader = LD3320SerialReader(args.port, args.baudrate).open()
    print('Listening. Press Ctrl+C to stop.')
    try:
        while True:
            data = reader.read_payload()
            if data:
                show_match(mapper, data)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        reader.close()


if __name__ == '__main__':
    main()
