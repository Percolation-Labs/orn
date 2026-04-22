# drafts/

Working area for blog-post drafts in Markdown. Not part of the released
package; lives on the `article-drafts` branch so main stays focused on code.

## Publishing to Medium (no API token needed)

The working pipeline is documented in `scripts/README.md` (local, git-ignored).
Short version: render the Markdown to HTML, commit it to this branch, and
paste the GitHub Pages URL into `medium.com/p/import`. The helper at
`scripts/publish_to_medium.sh <draft.md>` automates the render-commit-push
round trip and prints the URL to paste.

## Current draft

- `shared_coupling_is_a_world_model_hint.md` — the basic-ORN summary.
  Leads with the rotational invariance of per-layer coupling, frames
  parameter saving as the bonus, lands on the world-model / shared-ontology
  implication. Public page:
  <https://percolation-labs.github.io/orn/drafts/shared_coupling_is_a_world_model_hint.html>

## `.seed/` — voice / style reference (git-ignored)

Drop existing articles here as plain-text or Markdown to serve as a
writing-style reference. `.seed/` is in `.gitignore` so personal corpora
never get committed. One paragraph on what the folder is for is safe to
check in; everything else stays local.

Intended use: when drafting a new article, read a couple of files from
`.seed/` first to match voice (sentence rhythm, transitions, how
technical detail is set up before it lands). Do *not* paraphrase or
quote from them without attribution.
