#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO_BIN_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"

echo "Installing latent-ir from: $ROOT_DIR"
cargo install --path "$ROOT_DIR" --locked

if command -v latent-ir >/dev/null 2>&1; then
  echo "latent-ir available on PATH: $(command -v latent-ir)"
  latent-ir --help >/dev/null
  echo "Installation verified."
  exit 0
fi

echo "latent-ir is installed but not currently on PATH."
echo "Add this to your shell profile and restart shell:"
echo "  export PATH=\"$CARGO_BIN_DIR:\$PATH\""
echo "Or run in this shell now:"
echo "  export PATH=\"$CARGO_BIN_DIR:\$PATH\""
echo "Then verify with:"
echo "  latent-ir --help"
