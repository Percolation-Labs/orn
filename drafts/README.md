# drafts/

Working area for blog-post drafts in Markdown. Not part of the released
package; lives on the `article-drafts` branch so main stays focused on code.

## Publishing to Medium (no API token needed)

1. Write / edit the draft here as Markdown (one `.md` per article).
2. Upload it as a public Gist — `gh gist create --public <file>.md`.
3. Go to <https://medium.com/p/import> and paste the gist's raw URL.
4. Medium pulls the content into a new draft on your account (`@mrsirsh`).
5. Edit / polish in the Medium editor, publish when ready.

The helper at `../scripts/publish_to_medium.sh <draft.md>` wraps steps 2-3
— it creates the gist with `gh` and prints the raw URL plus the Medium
import URL to paste.

## Current drafts

- `shared_m_invariance.md` — a short explainer built around the COU-02
  motivating figure. Draft status: seed only.

## `.seed/` — voice / style reference (git-ignored)

Drop existing articles here as plain-text or Markdown to serve as a
writing-style reference. `.seed/` is in `.gitignore` so personal corpora
never get committed. One paragraph on what the folder is for is safe to
check in; everything else stays local.

Intended use: when drafting a new article, read a couple of files from
`.seed/` first to match voice (sentence rhythm, transitions, how
technical detail is set up before it lands). Do *not* paraphrase or
quote from them without attribution.
