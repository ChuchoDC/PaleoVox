#!/bin/bash
# ============================================================
#  PaleoVox macOS Build Script
#  Run this on a Mac with Python 3.8+ installed
# ============================================================

set -e

echo ""
echo "[1/4] Installing PyInstaller..."
pip3 install pyinstaller

echo ""
echo "[2/4] Installing project dependencies..."
pip3 install open3d numpy scipy matplotlib seaborn plotly scikit-learn PyQt5

echo ""
echo "[3/4] Building PaleoVox.app..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

pyinstaller --distpath "portable_versions/mac" \
    --workpath "/tmp/paleovox_build" \
    --clean \
    "portable_versions/PaleoVox.spec"

echo ""
echo "[4/4] Post-build cleanup..."

# Remove CUDA library to save space
rm -rf portable_versions/mac/PaleoVox/_internal/open3d/cuda 2>/dev/null || true

# Remove Dash (not needed by Plotly)
rm -rf portable_versions/mac/PaleoVox/_internal/dash* 2>/dev/null || true

# Remove unused matplotlib backends
rm -rf portable_versions/mac/PaleoVox/_internal/matplotlib/backends/backend_gtk* 2>/dev/null || true
rm -rf portable_versions/mac/PaleoVox/_internal/matplotlib/backends/backend_wx* 2>/dev/null || true
rm -rf portable_versions/mac/PaleoVox/_internal/matplotlib/backends/backend_tk* 2>/dev/null || true
rm -rf portable_versions/mac/PaleoVox/_internal/matplotlib/backends/backend_nbagg* 2>/dev/null || true
rm -rf portable_versions/mac/PaleoVox/_internal/matplotlib/backends/backend_pgf* 2>/dev/null || true

echo ""
echo "========================================"
echo "  Build complete!"
echo "  Output: portable_versions/mac/PaleoVox/"
echo "  Launch with: open PaleoVox/PaleoVox.bin"
echo "========================================"

# Create a launcher script
cat > "$SCRIPT_DIR/run_paleovox.command" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/PaleoVox/PaleoVox.bin" "$@"
LAUNCHER

chmod +x "$SCRIPT_DIR/run_paleovox.command"
echo "Launcher created: portable_versions/mac/run_paleovox.command"
