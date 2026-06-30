#!/usr/bin/env bash
# Start the TBJU GUI after restoring ELF2 headphone audio output.

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR" || exit 1

AUDIO_CARD="${TBJU_AUDIO_CARD:-rockchipnau8822}"
if ! amixer -c "$AUDIO_CARD" sget "PCM" >/dev/null 2>&1; then
    AUDIO_CARD="${TBJU_AUDIO_CARD_INDEX:-1}"
fi

amixer -c "$AUDIO_CARD" sset "PCM" 255 >/dev/null 2>&1 || true
amixer -c "$AUDIO_CARD" sset "Headphone" on >/dev/null 2>&1 || true
amixer -c "$AUDIO_CARD" sset "Headphone" 63 >/dev/null 2>&1 || true

# If you later use the SPKOUT 8-ohm speaker port instead of the 3.5mm jack,
# start this script as: TBJU_ENABLE_SPEAKER=1 ./start_gui_with_audio.sh
if [ "${TBJU_ENABLE_SPEAKER:-0}" = "1" ]; then
    amixer -c "$AUDIO_CARD" sset "Speaker" on >/dev/null 2>&1 || true
    amixer -c "$AUDIO_CARD" sset "Speaker" 63 >/dev/null 2>&1 || true
fi

pulseaudio --check >/dev/null 2>&1 || pulseaudio --start >/dev/null 2>&1 || true

unset TBJU_ALARM_ALSA_DEVICE
python3 run_gui.py

