#!/usr/bin/env bash
# Build script for the C++ hybrid chess engine (macOS / Linux)
# Usage: bash cpp/build.sh
#
# Prerequisites:
#   - g++ or clang++ with C++17 support
#   - pip install pybind11
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUT_DIR="$SCRIPT_DIR/../hybrid/cpp_engine"

# Detect compiler
if command -v g++ &>/dev/null; then
    CXX="g++"
elif command -v clang++ &>/dev/null; then
    CXX="clang++"
else
    echo "ERROR: No C++ compiler found. Install g++ or clang++." >&2
    exit 1
fi

# Get pybind11 includes and Python extension suffix
PY_INCLUDES=$(python3 -m pybind11 --includes 2>/dev/null || python -m pybind11 --includes)
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null \
          || python  -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

OUTPUT_FILE="$OUT_DIR/hybrid_cpp_engine${EXT_SUFFIX}"

echo "=== Building hybrid_cpp_engine ==="
echo "  Compiler: $CXX"
echo "  Sources:  board.cpp, rules.cpp, ab_search.cpp, bindings.cpp"
echo "  Output:   $OUTPUT_FILE"
echo ""

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Platform-specific flags
PLATFORM_FLAGS=""
case "$(uname -s)" in
    Darwin)
        # macOS: use -undefined dynamic_lookup to avoid linking Python at build time
        PLATFORM_FLAGS="-undefined dynamic_lookup"
        ;;
    Linux)
        # Linux: no special flags needed
        PLATFORM_FLAGS=""
        ;;
esac

$CXX -std=c++17 -O2 -Wall -shared -fPIC \
    $PLATFORM_FLAGS \
    $PY_INCLUDES \
    -o "$OUTPUT_FILE" \
    "$SRC_DIR/board.cpp" \
    "$SRC_DIR/rules.cpp" \
    "$SRC_DIR/ab_search.cpp" \
    "$SRC_DIR/bindings.cpp"

if [ $? -ne 0 ]; then
    echo "BUILD FAILED" >&2
    exit 1
fi

echo ""
echo "BUILD SUCCEEDED: $OUTPUT_FILE"

# Quick import test
echo ""
echo "Testing import..."
python3 -c "from hybrid.cpp_engine import hybrid_cpp_engine; print('  Import OK:', dir(hybrid_cpp_engine))" 2>/dev/null \
|| python -c "from hybrid.cpp_engine import hybrid_cpp_engine; print('  Import OK:', dir(hybrid_cpp_engine))"
