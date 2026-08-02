# Downstream language-modelling section

The write-up of the downstream evaluation, kept next to the artifacts it is generated from
rather than inside a paper directory, so it survives any given paper being replaced.

    downstream_section.tex        the section
    downstream_limitations.tex    the constraints paragraph, for a Limitations section
    generated/downstream_tables.tex   both tables, machine-written, do not edit

## Including it

Three lines in the main document:

```latex
\newcommand{\bnd}[1]{\texttt{bnd\_#1}}      % if not already defined
\newcommand{\downstreamdir}{path/to/marker_experiments/downstream/paper}
\input{\downstreamdir/downstream_section}
```

and, wherever Limitations lives:

```latex
\input{\downstreamdir/downstream_limitations}
```

`\downstreamdir` is the path from the **main** `.tex` to this directory, because LaTeX
resolves `\input` relative to the main document rather than to the including file. It
defaults to `.`, which is right only if the main document sits here.

Also needed: `\usepackage{booktabs}`, and one bibliography entry, `li2024datacomplm`
(DCLM, cited for the CORE metric). Nothing else from the old `custom.bib` is required.

## Regenerating the tables

```bash
uv run python marker_experiments/downstream/make_tex_tables.py
```

Reads three committed artifacts beside `make_tex_tables.py` (`manifest.json`,
`results.tsv`, `text_stats.json`) and rewrites `generated/downstream_tables.tex`. It works
from a clean clone with no arguments, and fails rather than emitting an empty table if an
artifact is missing. Every number in both tables comes from those files; none is typed.

During a sweep, point `--results` and `--text-stats` at the live copies under the
gitignored `results/` tree.
