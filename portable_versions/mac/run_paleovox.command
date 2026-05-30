#!/bin/bash
# PaleoVox macOS Launcher — double-click in Finder to run
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/PaleoVox/PaleoVox.bin" "$@"
