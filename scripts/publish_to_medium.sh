#!/usr/bin/env bash
# Render a Markdown draft to HTML, commit it to the current branch, push,
# wait for the GitHub Pages build, and print the Pages URL ready to paste
# into https://medium.com/p/import.
#
# Assumes: current branch has GitHub Pages enabled (source = this branch,
#          path = /).  One-time setup in scripts/README.md.
#
# Usage:  scripts/publish_to_medium.sh drafts/<article>.md
#
# Needs: python3 with the `markdown` package, `gh` (for pushing + API polling)
#        or plain git remote access.

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

stem=$(basename "$draft" .md)
html="drafts/${stem}.html"
pages_url="https://percolation-labs.github.io/orn/${html}"

echo "Rendering $draft → $html"
python3 - "$draft" "$html" <<'PY'
import sys, re, markdown
md_path, html_path = sys.argv[1], sys.argv[2]
md = open(md_path).read()
body = markdown.markdown(md, extensions=["fenced_code", "tables", "attr_list"])
m = re.match(r"<h1>(.*?)</h1>\s*", body)
title = m.group(1) if m else "Article"
body = body[m.end():] if m else body
open(html_path, "w").write(
    f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    f'<title>{title}</title>\n</head>\n<body>\n<article>\n<h1>{title}</h1>\n'
    f'{body}\n</article>\n</body>\n</html>\n'
)
PY

echo "Committing + pushing"
git add "$html"
git commit -m "Render ${stem} for Medium import" >/dev/null
git push >/dev/null 2>&1

echo "Waiting for GitHub Pages build"
for i in $(seq 1 40); do
  code=$(curl -sI "$pages_url" | head -1 | awk '{print $2}')
  if [ "$code" = "200" ]; then
    echo "  ready"
    break
  fi
  printf "  try %02d: HTTP %s\n" "$i" "$code"
  sleep 6
done

echo
echo "  URL (paste into https://medium.com/p/import):"
echo "    $pages_url"
echo
echo "Then open the URL above in a browser too and skim to confirm the content"
echo "looks right before importing on Medium."
