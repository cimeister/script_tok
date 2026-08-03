#!/usr/bin/env python3
"""Vocabulary-composition stats for the FineWeb 5GB matched grid, as a grid result file.

train_matched.py and train_multilang.py record compression numbers (chars/token,
roundtrip failures) in manifest.json as they train, but not the vocabulary-composition
diagnostics analyse_vocab() computes (duplicate-pair tax, marker-variant cost, word
coverage) -- those were only ever computed for the FineWiki multilang_grid.py sweep. This
loads the FineWeb 5GB tokenizers back from disk, runs the same analyse_vocab() over them,
and joins the result with manifest.json's compression numbers into one grid file with
multilang_result.json's schema, so make_intrinsic_table.py can read it like any other grid.

analyse_vocab is imported from multilang_grid.py rather than copied, so a change to the
definition (e.g. what counts as a "distinct alpha word") automatically applies here too
instead of silently drifting between two copies.

Every manifest field the output needs is read, never defaulted: a cell missing a required
field aborts the whole run naming the cell and the field, rather than writing a result
file with a guessed value baked into a column.

`lang` is a partial exception, and is called out here because it looks like the same rule
being broken: train_matched.py's cells (the four original fineweb_en_5gb_{plain,bnd_w,
bnd_wpd,bnd_wpd_caps}_bpe entries) predate the `lang` field train_multilang.py added, so
23 of the 27 cells carry `lang` in the manifest and 4 do not. But every cell, with or
without `lang`, carries `corpus`, and `corpus` has the fixed shape fineweb_{lang}_{size}
-- it is the same fact recorded in a different field of the same manifest record, not a
different or absent artifact. Reading `lang` out of `corpus` is therefore a parse of data
already in hand, not a substitution of a missing one: it is done for every cell (not just
the 4 missing `lang`), cross-checked against the manifest's own `lang` field wherever that
field exists, and printed at run time so a reader can see it happened rather than
inferring it silently. A corpus string that does not fit the fixed shape still aborts,
naming the cell and the corpus string, exactly like any other missing/malformed field.

    uv run python marker_experiments/downstream/vocab_stats.py

    # a subset, e.g. re-running after adding a language
    uv run python marker_experiments/downstream/vocab_stats.py --langs en,de
"""

import importlib
import json
import os
import re
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "py-nanochat"))

TOKENIZER_DIR = os.path.join(HERE, "tokenizers")
MANIFEST = os.path.join(REPO, "marker_experiments", "paper", "generated", "manifest.json")
DEFAULT_OUT = os.path.join(REPO, "marker_experiments", "fineweb5gb_result.json")

LANGS = ["en", "de", "fi", "ru", "ar"]
ARMS = ["plain", "bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_caps"]
TRAINERS = ["bpe", "mingram"]

# What every cell's manifest entry must carry to build one output row. Checked in full
# before anything is written, and any gap aborts -- see the module docstring. `lang` is
# NOT in this list: it is parsed from `corpus` for every cell instead (see CORPUS_RE
# below), because corpus is the field every cell actually carries.
REQUIRED_MANIFEST_FIELDS = [
    "arm", "trainer", "corpus",
    "total_vocab", "additional_vocab_size", "train_seconds",
    "eval_chars_per_token", "roundtrip_failures",
]
# Every corpus name used in this grid has this fixed shape: fineweb_{lang}_{size}, e.g.
# "fineweb_en_5gb". Parsing `lang` out of it is reading a fact the manifest already
# records under `corpus`, not inventing one -- see the module docstring.
CORPUS_RE = re.compile(r"^fineweb_(?P<lang>[a-z]{2,3})_(?P<size>[0-9a-z]+)$")
# Present for some grids' cells and not others (e.g. multilang_grid.py's FineWiki grid
# records unique_chunks and eval_docs; this FineWeb grid's manifest entries do not).
# Carried through when present, never invented when absent.
OPTIONAL_MANIFEST_FIELDS = ["eval_chars", "eval_docs", "eval_tokens", "unique_chunks"]

LOADER_DOTTED = {
    "bpe": "marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer",
    "mingram": "marker_experiments.downstream.boundary_tokenizer.BoundaryMinGramModel",
}

app = cyclopts.App()


def discover_cells(langs, arms, trainers, vocab, tokenizer_dir):
    """(lang, arm, trainer, corpus, path) for every combination with a tokenizer file.

    Not every (arm, trainer) combination was trained -- MinGram here only covers `plain`
    and `bnd_wpd` for English -- so this checks the filesystem rather than assuming the
    full cross product exists.
    """
    cells = []
    for lang in langs:
        corpus = f"fineweb_{lang}_5gb"
        for arm in arms:
            for trainer in trainers:
                path = os.path.join(tokenizer_dir, f"{corpus}_{arm}_{trainer}_v{vocab}.json.gz")
                if os.path.exists(path):
                    cells.append((lang, arm, trainer, corpus, path))
    return cells


@app.default
def main(
    langs: str = ",".join(LANGS),
    arms: str = ",".join(ARMS),
    trainers: str = ",".join(TRAINERS),
    vocab: int = 34685,
    tokenizer_dir: str = TOKENIZER_DIR,
    manifest: str = MANIFEST,
    out: str = DEFAULT_OUT,
) -> None:
    """Recompute vocabulary stats for every discovered cell and write one grid result file.

    Args:
        langs: Comma-separated languages. Each becomes corpus fineweb_{lang}_5gb.
        arms: Comma-separated pretokenizer arms to look for.
        trainers: Comma-separated trainers to look for.
        vocab: Matched total vocabulary (part of every tokenizer's filename).
        tokenizer_dir: Directory holding the {corpus}_{arm}_{trainer}_v{vocab}.json.gz files.
        manifest: manifest.json to read compression numbers from.
        out: Grid result JSON to write, in multilang_result.json's schema.
    """
    langs_l = [x.strip() for x in langs.split(",") if x.strip()]
    arms_l = [x.strip() for x in arms.split(",") if x.strip()]
    trainers_l = [x.strip() for x in trainers.split(",") if x.strip()]

    cells = discover_cells(langs_l, arms_l, trainers_l, vocab, tokenizer_dir)
    print(f"[vocab_stats] {len(cells)} tokenizer file(s) under {tokenizer_dir}:")
    for lang, arm, trainer, corpus, path in cells:
        print(f"  {os.path.basename(path)}")
    if not cells:
        raise SystemExit(f"no tokenizer files matched under {tokenizer_dir} for the given langs/arms/trainers")

    with open(manifest) as f:
        manifest_data = json.load(f)

    # Validate every cell before loading a single tokenizer: a run that fails halfway
    # through, after already having reloaded a dozen multi-GB-adjacent tokenizer files,
    # is a worse failure mode than one that fails immediately with the full list of gaps.
    problems = []
    parsed_lang = {}          # key -> lang, parsed from `corpus`, reused in the 2nd pass
    lang_from_manifest = 0    # counts for the run-time line the docstring promises
    lang_from_corpus_only = 0
    for lang, arm, trainer, corpus, path in cells:
        key = f"{corpus}_{arm}_{trainer}_v{vocab}"
        cell = manifest_data.get(key)
        if cell is None:
            problems.append(f"{key}: not present in {manifest}")
            continue
        for field in REQUIRED_MANIFEST_FIELDS:
            if field not in cell:
                problems.append(f"{key}: missing required manifest field {field!r}")
        # Cross-check rather than trust one source: the filename and the manifest entry
        # should agree on arm/trainer/corpus. A mismatch means either the manifest was
        # merged from the wrong part file or a tokenizer was renamed after training,
        # either of which would otherwise silently mislabel a row in the output grid.
        for field, expected in (("arm", arm), ("trainer", trainer), ("corpus", corpus)):
            if field in cell and cell[field] != expected:
                problems.append(
                    f"{key}: manifest {field}={cell[field]!r} disagrees with filename-derived {expected!r}"
                )
        # `lang`: parsed from `corpus` (required above, so already checked present) for
        # every cell, then cross-checked against the manifest's own `lang` field wherever
        # that field exists. See CORPUS_RE and the module docstring.
        if "corpus" in cell:
            m = CORPUS_RE.match(cell["corpus"])
            if not m:
                problems.append(
                    f"{key}: corpus {cell['corpus']!r} does not match the fixed "
                    "fineweb_{lang}_{size} shape; cannot parse lang"
                )
            else:
                parsed_lang[key] = m.group("lang")
                if "lang" in cell:
                    lang_from_manifest += 1
                    if cell["lang"] != parsed_lang[key]:
                        problems.append(
                            f"{key}: manifest lang={cell['lang']!r} disagrees with "
                            f"corpus-parsed lang={parsed_lang[key]!r} (corpus={cell['corpus']!r})"
                        )
                else:
                    lang_from_corpus_only += 1
    if problems:
        raise SystemExit(
            f"[vocab_stats] {len(problems)} manifest problem(s); aborting rather than "
            "substituting a default or dropping the cell:\n" + "\n".join(f"  {p}" for p in problems)
        )
    print(f"[vocab_stats] lang: {lang_from_manifest} cell(s) had a manifest 'lang' field "
          f"(cross-checked against corpus), {lang_from_corpus_only} cell(s) had lang "
          "parsed from corpus alone (no 'lang' field in manifest)")

    # Both imports register pretokenizer subclasses as a side effect (see
    # boundary_tokenizer.py's docstring) and must happen before any .load() call.
    loader_cls = {}
    for trainer in set(trainers_l):
        module_name, _, cls_name = LOADER_DOTTED[trainer].rpartition(".")
        loader_cls[trainer] = getattr(importlib.import_module(module_name), cls_name)

    # Imported lazily via importlib, not `from marker_experiments.multilang_grid import
    # analyse_vocab` at module scope, so that importing multilang_grid (which pulls in
    # script_bpe.corpus.base and the scriptenc_marker_v4/v5 pretokenizers) only happens
    # once this script is actually run, not on every --help.
    analyse_vocab = importlib.import_module("marker_experiments.multilang_grid").analyse_vocab

    results = {}
    for lang, arm, trainer, corpus, path in cells:
        key = f"{corpus}_{arm}_{trainer}_v{vocab}"
        cell = manifest_data[key]
        tokenizer = loader_cls[trainer].load(path)
        stats = analyse_vocab(tokenizer, tokenizer.pretokenizer)

        out_key = f"{lang}_{arm}_{trainer}"
        entry = {
            "lang": parsed_lang[key],
            "pretokenizer": cell["arm"],
            "method": cell["trainer"],
            "additional_vocab_size": cell["additional_vocab_size"],
            "vocab_size": cell["total_vocab"],
            "train_seconds": cell["train_seconds"],
        }
        for field in OPTIONAL_MANIFEST_FIELDS:
            if field in cell:
                entry[field] = cell[field]
        entry["eval_chars_per_token"] = cell["eval_chars_per_token"]
        entry["roundtrip_failures"] = cell["roundtrip_failures"]
        entry.update(stats)
        results[out_key] = entry
        print(f"  {out_key}: {cell['eval_chars_per_token']:.4f} ch/tok  "
              f"rt_fail={cell['roundtrip_failures']}  "
              f"dup_pairs={stats['space_dup_pairs']}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[vocab_stats] {len(results)} cell(s) -> {out}")


if __name__ == "__main__":
    app()
