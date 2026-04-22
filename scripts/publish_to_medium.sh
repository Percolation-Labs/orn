#!/usr/bin/env bash
# Bridge a Markdown draft to Medium via the gist→import pipeline.
#
# Usage:  scripts/publish_to_medium.sh drafts/shared_m_invariance.md
#
# Needs `gh` (GitHub CLI) authenticated.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <draft.md>" >&2
  exit 1
fi

draft="$1"
if [ ! -f "$draft" ]; then
  echo "No such file: $draft" >&2
  exit 1
fi

title=$(head -n 1 "$draft" | sed 's/^# *//')

echo "Creating public gist for: $title"
gist_url=$(gh gist create --public --desc "$title" "$draft")
gist_id=$(basename "$gist_url")

# The raw URL for a single-file gist.
raw_url="https://gist.githubusercontent.com/$(gh api user --jq .login)/${gist_id}/raw/$(basename "$draft")"

echo
echo "  Gist:        $gist_url"
echo "  Raw URL:     $raw_url"
echo
echo "Next: open https://medium.com/p/import and paste the raw URL above."
echo "  Medium pulls the Markdown into a new draft on your account."
