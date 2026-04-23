# arXiv submission checklist (ORN paper)

Source lives next to this file:

- `the_orbital_response_network.tex` — the paper source.
- `orn_v3_scaling.pdf` — the single external figure the tex `\includegraphics` pulls in.
- `orn_arxiv_submission.tar.gz` — prebuilt submission tarball (top-level tex + figure, no nesting).

## Target categories

- **Primary:** `cs.CL` (Computation and Language).
- **Cross-list:** `cs.LG` (Machine Learning). Optionally `cs.NE`.

## Metadata to paste into the arXiv web form

- **Title:** *The Orbital Response Network: Shared Coupling Geometry as an Alternative to Per-Layer Attention*
- **Authors:** *Sirsh Amarteifio (Percolation Labs)*
- **Abstract:** copy from the tex `abstract` environment (1858 chars, under the 1920 soft limit).
- **Comments:** *Code and published checkpoints at https://github.com/Percolation-Labs/orn and https://huggingface.co/mr-saoirse/orn-v3-605m*
- **License:** choose one deliberately.
  - `arXiv.org perpetual, non-exclusive license` is the default and works with every venue.
  - `CC BY 4.0` if you want free reuse. Check that no venue you plan to submit to forbids it.
- **MSC/ACM classes:** leave blank.

## One-time setup

- Register an arXiv account at <https://arxiv.org/user/register> using a real name and, ideally, an institutional email (Percolation Labs).
- **Endorsement.** If this is your first `cs.CL` / `cs.LG` submission and you do not already have papers under those categories, you need an endorsement from someone with 3+ papers in that category in the last 5 years. Send them the endorsement link from <https://arxiv.org/auth/need-endorsement>. The endorser clicks one link.

## Submit

1. Go to <https://arxiv.org/submit>.
2. Upload `orn_arxiv_submission.tar.gz`.
3. arXiv will build the PDF from source on their TeX Live. Review the build log and the rendered PDF in their viewer.
4. Fill in metadata from the list above.
5. Review, submit.
6. Expect announcement the next business day (~20:00 ET). First-time submissions can be held briefly for moderation.

## Announcement cadence

- Submit **before 14:00 ET Mon–Fri** → announced the next business day (~20:00 ET).
- After that cut-off you skip a day.

## After announcement

- Update `born/paper/SUBMISSION.md` with the arXiv ID (e.g. `2604.12345`).
- Add a link to the arXiv preprint in the repo README and Medium article.
- Optional: post the arXiv link on the Percolation Labs channels.

## Local dry-run build

BasicTeX is small enough to be worth installing for a local sanity build:

```bash
brew install --cask basictex          # ~100 MB
eval "$(/usr/libexec/path_helper)"    # pick up new tex tools
sudo tlmgr update --self
sudo tlmgr install natbib hyperref amsmath pgfplots tikz booktabs \
                    xcolor microtype wrapfig subcaption
cd born/paper
pdflatex the_orbital_response_network.tex
pdflatex the_orbital_response_network.tex   # second pass for cross-refs
```

If this compiles without missing-file errors, the tarball will compile on arXiv too.

## Rebuilding the submission tarball

```bash
cd /tmp && rm -rf orn-arxiv && mkdir orn-arxiv && cd orn-arxiv
cp /Users/sirsh/code/X/ai/orn/born/paper/the_orbital_response_network.tex .
cp /Users/sirsh/code/X/ai/orn/born/paper/orn_v3_scaling.pdf .
tar czf /Users/sirsh/code/X/ai/orn/born/paper/orn_arxiv_submission.tar.gz \
    the_orbital_response_network.tex orn_v3_scaling.pdf
```

## Common first-submission gotchas

- Any `\url` or hyperlink that crosses a line break can silently break arXiv's bib linker — wrap long URLs in `\url{}` (we already do).
- Footnotes inside the abstract are not allowed by arXiv; we do not use any.
- arXiv builds at a specific TeX Live date; packages newer than that date are not available. Our `.tex` only uses classic packages so this is not a risk.
- Figure files must be lowercase + simple names. `orn_v3_scaling.pdf` is fine.
- `natbib` is our bib style — keep it. Some CS venues prefer `plainnat`, which is already what the tex sets.
