#!/bin/sh
# Scope 3 Auditor installer for macOS and Linux. Syntax checked; Windows is the tested path.
set -eu

REPO_URL=""
BRANCH="main"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
TARGET="${SCOPE3_SOURCE:-${SCOPE3_DIR:-$SCRIPT_DIR}}"
if [ -f "$SCRIPT_DIR/PUBLIC-DEMO-MODE" ]; then SCOPE3_SOURCE="$SCRIPT_DIR"; TARGET="$SCRIPT_DIR"; fi
UV_VERSION="0.12.13"
TIER="${SCOPE3_TIER:-check}"

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
valid_source() { [ -f "$1/start.py" ] && [ -f "$1/requirements.txt" ]; }
valid_origin() {
  command -v git >/dev/null 2>&1 || die "'$1' already exists. Set SCOPE3_SOURCE only after reviewing that local source."
  origin="$(git -C "$1" remote get-url origin 2>/dev/null || true)"
  [ "$origin" = "$REPO_URL" ] || [ "$origin" = "$REPO_URL.git" ] ||
    die "'$1' already exists but its Git origin is not $REPO_URL. Set SCOPE3_SOURCE only after reviewing that local source."
}
case "$TIER" in browse|check|full) ;; *) die "SCOPE3_TIER must be browse, check or full." ;; esac

if [ -z "${SCOPE3_SOURCE:-}" ]; then
  if [ -e "$TARGET" ]; then
    valid_source "$TARGET" || die "'$TARGET' is not a Scope 3 Auditor checkout."
    valid_origin "$TARGET"
  else
    command -v git >/dev/null 2>&1 || die "Git is required for a fresh download, or set SCOPE3_SOURCE to an existing checkout."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TARGET"
  fi
fi
valid_source "$TARGET" || die "'$TARGET' is not a Scope 3 Auditor checkout."
TARGET="$(cd "$TARGET" && pwd -P)"
TOOLS="$TARGET/.tools"
mkdir -p "$TOOLS"

UV="${SCOPE3_UV:-$(command -v uv 2>/dev/null || true)}"
if [ -z "$UV" ]; then
  die "uv $UV_VERSION is required. Install uv from https://docs.astral.sh/uv/getting-started/installation/, then run this installer again."
fi
[ -x "$UV" ] || die "uv installation failed."
UV_FOUND="$("$UV" --version)"
set -- $UV_FOUND
[ "${1:-}" = "uv" ] && [ "${2:-}" = "$UV_VERSION" ] || die "uv $UV_VERSION is required; found '$UV_FOUND'."
if [ "$UV" != "$TOOLS/uv" ]; then cp "$UV" "$TOOLS/uv"; chmod +x "$TOOLS/uv"; fi
UV="$TOOLS/uv"

PYTHON_DIR="$TOOLS/python"
"$UV" python install 3.12 --install-dir "$PYTHON_DIR" --no-bin
export UV_PYTHON_INSTALL_DIR="$PYTHON_DIR"
PYTHON="$("$UV" python find 3.12 --managed-python)"
[ -n "$PYTHON" ] || die "Python 3.12 could not be found after installation."

export SCOPE3_UV="$UV"
cd "$TARGET"
if [ "$TIER" != "browse" ]; then "$PYTHON" start.py --install; fi
if [ "${SCOPE3_NO_START:-}" = "1" ]; then exit 0; fi
if [ "$TIER" = "browse" ]; then exec "$PYTHON" start.py --browse; fi
exec "$PYTHON" start.py
