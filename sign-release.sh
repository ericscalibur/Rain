#!/bin/bash
# sign-release.sh — Create a signed, checksummed release archive
#
# Usage:
#   ./sign-release.sh [version] [files/dirs to include...]
#
# Examples:
#   ./sign-release.sh v1.0.0 src/ README.md requirements.txt
#   ./sign-release.sh v2.1.3 dist/ LICENSE
#
# Requirements:
#   - gpg (gnupg) installed
#   - A GPG key in your keychain
#   - git (for repo name + tag detection)
#
# Optional setup:
#   Set a default signing key in git config:
#     git config --global user.signingkey YOUR_KEY_ID
#
#   Set your public key URL (shown in verification instructions):
#     git config --global user.pgpkeyurl https://keybase.io/yourname/pgp_keys.asc
#
# MIT License — https://opensource.org/licenses/MIT

set -e

# ── Resolve GPG key ───────────────────────────────────────────────────────────
GPG_KEY=$(git config --global user.signingkey 2>/dev/null || true)

if [ -z "${GPG_KEY}" ]; then
  echo "No signing key configured. Options:"
  echo "  1. Set globally:  git config --global user.signingkey YOUR_KEY_ID"
  echo "  2. Set locally:   git config user.signingkey YOUR_KEY_ID"
  echo ""
  echo "To list your available keys:"
  echo "  gpg --list-secret-keys --keyid-format LONG"
  exit 1
fi

# ── Resolve public key URL (optional, for verification instructions) ──────────
PGP_KEY_URL=$(git config --global user.pgpkeyurl 2>/dev/null || true)

# ── Resolve repo name + version ───────────────────────────────────────────────
REPO_NAME=$(basename "$(git rev-parse --show-toplevel 2>/dev/null || echo "release")")
VERSION="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo 'v0.0.1')}"
RELEASE_DIR="releases"
ARCHIVE="${REPO_NAME}-${VERSION}.tar.gz"
SIG="${ARCHIVE}.asc"
CHECKSUM="${ARCHIVE}.sha256"

# ── Files to include ──────────────────────────────────────────────────────────
shift || true
INCLUDE=("$@")

if [ ${#INCLUDE[@]} -eq 0 ]; then
  echo "Usage: ./sign-release.sh [version] [files/dirs to include...]"
  echo "Example: ./sign-release.sh v1.0.0 src/ README.md requirements.txt"
  exit 1
fi

# ── Build archive ─────────────────────────────────────────────────────────────
echo "📦 Building ${ARCHIVE}..."
mkdir -p "${RELEASE_DIR}"
tar -czf "${RELEASE_DIR}/${ARCHIVE}" "${INCLUDE[@]}"

# ── Sign ──────────────────────────────────────────────────────────────────────
echo "✍️  Signing with key ${GPG_KEY}..."
export GPG_TTY=$(tty)
gpg --batch --yes \
    --local-user "${GPG_KEY}" \
    --detach-sign --armor \
    "${RELEASE_DIR}/${ARCHIVE}"

# ── Checksum ──────────────────────────────────────────────────────────────────
shasum -a 256 "${RELEASE_DIR}/${ARCHIVE}" > "${RELEASE_DIR}/${CHECKSUM}"
echo "📋 SHA256: $(cat "${RELEASE_DIR}/${CHECKSUM}" | awk '{print $1}')"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "✅ Release ready in ./${RELEASE_DIR}/"
echo "   ${ARCHIVE}"
echo "   ${SIG}"
echo "   ${CHECKSUM}"
echo ""

# ── Verification instructions ─────────────────────────────────────────────────
echo "── Paste into your release notes ────────────────────────────────────────"
echo "## Verifying authenticity"
echo ""
if [ -n "${PGP_KEY_URL}" ]; then
echo "Import the signing key:"
echo '```bash'
echo "curl ${PGP_KEY_URL} | gpg --import"
echo '```'
echo ""
fi
echo "Verify the signature:"
echo '```bash'
echo "gpg --verify ${SIG} ${ARCHIVE}"
echo '```'
echo ""
echo "Verify the checksum:"
echo '```bash'
echo "shasum -a 256 -c ${CHECKSUM}"
echo '```'
echo ""
echo "A message of \"Good signature from...\" confirms the release is authentic."
echo "─────────────────────────────────────────────────────────────────────────"
