#!/usr/bin/env sh
set -e

PROJECT_NAME="mindforge"
INSTALL_DIR="$HOME/.${PROJECT_NAME}-venv"
SHIM_DIR="$HOME/.local/bin"
SHIM_PATH="$SHIM_DIR/${PROJECT_NAME}"
SRC_DIR="$HOME/.${PROJECT_NAME}-src"
PY_MIN_MAJOR=3
PY_MIN_MINOR=9
MARK_BEGIN="# ${PROJECT_NAME} PATH BEGIN"
MARK_END="# ${PROJECT_NAME} PATH END"

info() { printf "[+] %s\n" "$1"; }
warn() { printf "[!] %s\n" "$1" >&2; }
err() { printf "[x] %s\n" "$1" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<EOF
${PROJECT_NAME} installer

Usage: curl -fsSL <URL>/install.sh | sh [-s -- [--uninstall]]
       ./install.sh [--uninstall]

Options:
  --uninstall   Remove venv, shim, and PATH entries
  --prefix DIR  Install shim under DIR/bin instead of ~/.local/bin
  --verbose     Print extra logs
EOF
}

VERBOSE=0
UNINSTALL=0
PREFIX=""
FORCE_DEPS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall) UNINSTALL=1 ;;
    --prefix) shift; PREFIX="$1" ;;
    --verbose) VERBOSE=1 ;;
    --force-deps) FORCE_DEPS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) warn "unknown arg: $1" ;;
  esac
  shift
done

[ "$VERBOSE" -eq 1 ] && set -x

if [ "$(id -u)" = "0" ]; then
  err "Do not run as root"
fi

# Determine shim dir
if [ -n "$PREFIX" ]; then
  SHIM_DIR="$PREFIX/bin"
  SHIM_PATH="$SHIM_DIR/${PROJECT_NAME}"
fi

mkdir -p "$SHIM_DIR"

# Uninstall path
if [ "$UNINSTALL" -eq 1 ]; then
  info "Uninstalling ${PROJECT_NAME}"
  rm -rf "$INSTALL_DIR" "$SRC_DIR"
  if [ -f "$SHIM_PATH" ]; then rm -f "$SHIM_PATH"; fi
  # Remove PATH markers from shell configs
  for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.config/fish/config.fish"; do
    [ -f "$rc" ] || continue
    tmp="$rc.tmp.$$"
    awk "/^${MARK_BEGIN}$/,/^${MARK_END}$/ {next} {print}" "$rc" > "$tmp" && mv "$tmp" "$rc"
  done
  info "Uninstalled"
  exit 0
fi

# Ensure curl or wget
if have_cmd curl; then
  DL="curl -fsSL"
elif have_cmd wget; then
  DL="wget -qO-"
else
  err "Need curl or wget"
fi

# Ensure Python
if have_cmd python3; then PY=python3
elif have_cmd python; then PY=python
else err "Python 3 not found"; fi

# Version check
PYV=$($PY - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
PYMAJOR=$(printf "%s" "$PYV" | cut -d. -f1)
PYMINOR=$(printf "%s" "$PYV" | cut -d. -f2)
if [ "$PYMAJOR" -lt "$PY_MIN_MAJOR" ] || { [ "$PYMAJOR" -eq "$PY_MIN_MAJOR" ] && [ "$PYMINOR" -lt "$PY_MIN_MINOR" ]; }; then
  err "Python >= ${PY_MIN_MAJOR}.${PY_MIN_MINOR} required (found $PYV)"
fi

# Create venv
if [ ! -f "$INSTALL_DIR/bin/activate" ]; then
  info "Creating virtualenv at $INSTALL_DIR"
  "$PY" -m venv "$INSTALL_DIR"
fi
. "$INSTALL_DIR/bin/activate"

# Upgrade pip
python -m pip -q install --upgrade pip

# Install package from local path if present, else fallback to repo archive placeholder
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$ROOT_DIR/setup.py" ] || [ -f "$ROOT_DIR/pyproject.toml" ]; then
  info "Installing ${PROJECT_NAME} from local source"
  if [ -f "$ROOT_DIR/requirements.txt" ]; then pip -q install -r "$ROOT_DIR/requirements.txt"; fi
  pip -q install -e "$ROOT_DIR"
else
  GIT_URL_DEFAULT="https://github.com/Exw27/mindforge.git"
  GIT_URL="${GIT_URL:-$GIT_URL_DEFAULT}"
  info "Installing ${PROJECT_NAME} from Git: $GIT_URL"
  pip -q install --upgrade pip setuptools wheel
  pip -q install "git+${GIT_URL}#egg=${PROJECT_NAME}"
  if [ "$FORCE_DEPS" -eq 1 ] || ! python -c 'import transformers,fastapi,uvicorn,torch' >/dev/null 2>&1; then
    info "Installing runtime dependencies"
    pip -q install transformers fastapi uvicorn torch tqdm llama-cpp-python huggingface-hub
  fi
fi

# Create shim
cat > "$SHIM_PATH" <<'SH'
#!/usr/bin/env sh
VENV="$HOME/.mindforge-venv"
if [ -n "$MINDFORGE_VENV" ]; then VENV="$MINDFORGE_VENV"; fi
exec "$VENV/bin/python" -m mindforge.main "$@"
SH
chmod +x "$SHIM_PATH"

# Ensure ~/.local/bin (or prefix) on PATH in shell rc files
add_path_snippet() {
  rc="$1"
  [ -f "$rc" ] || touch "$rc"
  if ! grep -q "^${MARK_BEGIN}$" "$rc" 2>/dev/null; then
    {
      echo "$MARK_BEGIN"
      if printf "%s" "$rc" | grep -q "fish/config.fish$"; then
        echo "if test -d $SHIM_DIR; and not contains $SHIM_DIR $PATH; fish_add_path $SHIM_DIR; end"
      else
        echo "if [ -d $SHIM_DIR ]; then PATH=\"$SHIM_DIR:$PATH\"; export PATH; fi"
        echo "alias ${PROJECT_NAME}=\"$SHIM_PATH\""
      fi
      echo "$MARK_END"
    } >> "$rc"
  fi
}

add_path_snippet "$HOME/.bashrc"
add_path_snippet "$HOME/.zshrc"
mkdir -p "$HOME/.config/fish"
add_path_snippet "$HOME/.config/fish/config.fish"

info "Installed ${PROJECT_NAME}"
info "Open a new shell or run: export PATH=\"$SHIM_DIR:$PATH\""
info "Then run: ${PROJECT_NAME} --help"
