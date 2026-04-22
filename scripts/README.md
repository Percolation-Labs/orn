# scripts/

Local-only helpers for the drafts workflow. The whole folder is git-ignored
(see root `.gitignore`) so nothing here is pushed.

## publish_to_medium.sh

End-to-end: Markdown file in `drafts/` to a Medium-importable URL.

```
scripts/publish_to_medium.sh drafts/three_plots_in_a_minute.md
```

Prints a URL like
`https://percolation-labs.github.io/orn/drafts/three_plots_in_a_minute.html`.
Paste that into `https://medium.com/p/import` while signed in as @mrsirsh.
Medium fetches the page, extracts the article, and creates a draft on your
account.

What the script does, in order:

1. Renders the Markdown to a standalone HTML file under `drafts/<stem>.html`
   using `python3 -m markdown` (`fenced_code`, `tables`, `attr_list` extensions).
   The first `# Heading` becomes the HTML `<title>` and `<h1>`; the rest of
   the body is wrapped in an `<article>` element.
2. Commits and pushes the HTML file to the current branch.
3. Polls `https://percolation-labs.github.io/orn/drafts/<stem>.html` every
   six seconds until it returns HTTP 200. First build usually takes about
   thirty seconds.
4. Prints the Pages URL.

## Why this path and not something simpler

Things that look obvious but do not work:

- **Gist raw URL (.md).** Serves as `text/plain`. Medium's importer silently
  drops `text/plain` content and returns nothing useful.
- **Gist raw URL (.html).** Still served as `text/plain` because gists sandbox
  every file. Same outcome.
- **htmlpreview.github.io wrapper.** The wrapper page is a JavaScript loader
  that fetches the raw file client-side. Medium's importer is server-side and
  only sees an empty shell.
- **GitHub blob view of `.md` (`github.com/.../blob/branch/file.md`).** Served
  as `text/html` with the rendered Markdown inside, but buried in a lot of
  chrome (nav, sidebars, file tree). Medium's extractor often grabs the wrong
  region.

What works:

- **GitHub Pages serving real HTML.** The article HTML is a minimal
  `<article>` wrapper with no extra chrome, served as `text/html` directly
  by GitHub Pages. Medium's extractor lands on the `<article>` cleanly.

## One-time setup

Enable GitHub Pages on the branch where the HTML will live. For this repo,
Pages is enabled on `article-drafts` (source: branch `article-drafts`,
path `/`). To enable elsewhere:

```
gh api "repos/<owner>/<repo>/pages" -X POST \
   -f "source[branch]=<branch>" -f "source[path]=/"
```

The script assumes `https://percolation-labs.github.io/orn/` as the Pages
root. If the repo moves or the branch changes, update the `pages_url`
prefix in the script.

## Images

Markdown image links need to be absolute URLs so Medium can fetch them on
import. The convention used in `drafts/three_plots_in_a_minute.md` is
`https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/<file>.png`,
which works on the public repo without auth. Keep the PNGs committed to the
branch (they are public anyway once pushed).

## When Medium's import still fails

Medium's import tool is known-flaky. If a URL that looks fine gets refused:

1. Open the Pages URL in a browser to confirm the content renders correctly.
   If you see the article, the URL is good and Medium is at fault.
2. Retry the import a minute later. The first attempt sometimes fails
   silently on transient fetch errors.
3. Fall back to manual paste: open `https://medium.com/new-story`, paste
   the raw Markdown from `drafts/<article>.md`, and upload images from
   `drafts/figs/` by drag-and-drop. Medium's editor parses Markdown
   headings, bold/italic, and code blocks on paste.

## Voice / style references

`drafts/.seed/` (also git-ignored) holds existing articles from the target
author's Medium profile for style reference. Drop `.md` or `.txt` files
there when drafting a new piece.
