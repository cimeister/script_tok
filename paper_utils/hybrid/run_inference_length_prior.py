#!/usr/bin/env python3
"""Stage 0: effect of an inference-time geometric length prior on the paper's Unigram tokenizers.

Subtracting a constant lambda from every token log_prob penalises each emitted token by lambda,
i.e. a geometric length prior on the segmentation. No training is involved: for each language
and lambda we load the default Unigram model, bias it in place, and measure compression on the
Goldfish eval corpus (plus MorphAlign for eng/deu/fin).
"""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from paper_utils.hybrid.generate_morphalign_scatter import (
    ALIGN_MODULE,
    LANGUAGE_CONFIGS as MORPHALIGN_LANGUAGE_CONFIGS,
    MORPHALIGN_ITERATIONS,
    MORPHALIGN_METRIC_NAME,
    MORPHALIGN_MODEL,
    MORPHALIGN_THRESHOLDS,
    _write_segmented_tsv,
)
from paper_utils.hybrid.utils import (
    TRAIN_CORPUS_SUFFIX,
    apply_inference_bias,
    geomean,
    load_cache,
    morphalign_paper_score,
    paper_table_path,
    save_cache,
)
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.unigram import UnigramModel

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
RESULTS_DIR = Path("results/hybrid")
CACHE_PATH = RESULTS_DIR / "cache_inference_length_prior.json"
GRID_CACHE_PATH = RESULTS_DIR / "cache_train_eval_compression_grid.json"
SEGMENTED_DIR = RESULTS_DIR / "morphalign_segmented_inference_prior"
TABLE_STEM = "table_inference_length_prior"
EXPECTED_TRAIN_SUFFIX = "_quick"
EXPECTED_PRETOKENIZER_HASH = "PT-ec426101"
DEFAULT_MAX_WORKERS = 30
BASELINE_LAMBDA = 0.0

# (short code used by FineWeb names, eval-corpus language code, label)
LANGUAGES = [
    ("en", "eng", "English"),
    ("de", "deu", "German"),
    ("fi", "fin", "Finnish"),
    ("ru", "rus", "Russian"),
    ("ar", "arb", "Arabic"),
    ("ko", "kor", "Korean"),
]
EVAL_CORPORA = {
    "eng": "eng_latn_fishfood",
    "deu": "deu_latn_fishfood",
    "fin": "fin_latn_fishfood",
    "rus": "rus_cyrl_fishfood",
    "arb": "arb_arab_fishfood",
    "kor": "kor_hang_fishfood",
}
GOLD_FILES = {cfg["lang"]: Path(cfg["gold_file"]) for cfg in MORPHALIGN_LANGUAGE_CONFIGS}
MORPHALIGN_LANGS = ["eng", "deu", "fin"]


def train_corpus(short: str) -> str:
    return f"fineweb_{short}_5gb{TRAIN_CORPUS_SUFFIX}"


def cache_key(train: str, eval_corpus: str, lam: float) -> str:
    return f"{train}/{eval_corpus}/lambda{lam}"


def model_path(lang3: str) -> Path:
    short = next(s for s, l3, _ in LANGUAGES if l3 == lang3)
    return get_unigram_model_path(train_corpus(short), DEFAULTS)


def build_tasks() -> list[dict]:
    tasks = []
    for short, lang3, _label in LANGUAGES:
        for lam in LAMBDAS:
            tasks.append(
                {
                    "lang": lang3,
                    "lam": lam,
                    "train_corpus": train_corpus(short),
                    "eval_corpus": EVAL_CORPORA[lang3],
                    "model_path": str(model_path(lang3)),
                    "cache_key": cache_key(train_corpus(short), EVAL_CORPORA[lang3], lam),
                }
            )
    return tasks


def check_preconditions() -> None:
    """Fail before any heavy work if a model or cached eval corpus is missing, or the pretokenizer differs."""
    if TRAIN_CORPUS_SUFFIX != EXPECTED_TRAIN_SUFFIX:
        raise RuntimeError(f"MINGRAM_TRAIN_CORPUS_SUFFIX={TRAIN_CORPUS_SUFFIX!r}, expected {EXPECTED_TRAIN_SUFFIX!r}")
    for lang3 in MORPHALIGN_LANGS:
        if not GOLD_FILES[lang3].exists():
            raise FileNotFoundError(f"MorphAlign gold file missing: {GOLD_FILES[lang3]}")
    for short, lang3, _label in LANGUAGES:
        path = model_path(lang3)
        if not path.exists():
            raise FileNotFoundError(f"Unigram model missing: {path}")
        model = UnigramModel.load(str(path))
        got = model.pretokenizer.hash()
        if got != EXPECTED_PRETOKENIZER_HASH:
            raise RuntimeError(f"{path}: pretokenizer hash {got}, expected {EXPECTED_PRETOKENIZER_HASH}")
        meta = PretokenizedCorpus(
            EVAL_CORPORA[lang3], PretokenizedCorpus.DEFAULT_BASE_PATH, model.pretokenizer, dummy=True
        ).metadata_path()
        if not os.path.exists(meta):
            raise FileNotFoundError(
                f"Eval corpus {EVAL_CORPORA[lang3]} not cached for pretokenizer {got}: {meta} "
                "(refusing to let load_corpus_by_name build it)"
            )
        del model


def run_task(task: dict) -> tuple[str, dict]:
    model = UnigramModel.load(task["model_path"])  # fresh model: the bias is applied in place
    assert model.pretokenizer.hash() == EXPECTED_PRETOKENIZER_HASH
    apply_inference_bias(model, task["lam"])
    eval_corpus = load_corpus_by_name(task["eval_corpus"], model.pretokenizer)
    perf = model.corpus_performance(eval_corpus)
    result = {
        "tokens": int(perf["total_tokens_len"]),
        "bytes": int(perf["total_byte_len"]),
        "morphalign": None,
    }
    lang = task["lang"]
    if lang in MORPHALIGN_LANGS:
        gold_file = GOLD_FILES[lang]
        out_tsv = SEGMENTED_DIR / f"{lang}_lambda{task['lam']}.tsv"
        _write_segmented_tsv(model, gold_file, out_tsv)  # encodes with the biased model
        results, _name = ALIGN_MODULE.evaluate_segmentations(
            str(gold_file),
            str(out_tsv),
            MORPHALIGN_THRESHOLDS,
            MORPHALIGN_ITERATIONS,
            MORPHALIGN_MODEL,
            skip_gold_train=True,
        )
        result["morphalign"] = float(results[MORPHALIGN_METRIC_NAME])
    return task["cache_key"], result


def compute(tasks: list[dict], cache: dict, max_workers: int) -> dict:
    missing = [t for t in tasks if t["cache_key"] not in cache]
    print(f"Tasks: total={len(tasks)} missing={len(missing)}", flush=True)
    if not missing:
        return cache
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(run_task, t): t for t in missing}
        for done, future in enumerate(as_completed(futures), 1):
            key, result = future.result()  # raises if a worker failed
            cache[key] = result
            save_cache(cache, CACHE_PATH)
            print(f"[{done}/{len(missing)}] {key}: {result}", flush=True)
    return cache


def crosscheck_baseline(cache: dict) -> bool:
    grid = load_cache(GRID_CACHE_PATH)
    ok = True
    for short, lang3, _label in LANGUAGES:
        train = train_corpus(short)
        path = model_path(lang3)
        grid_key = f"tokens/{train}/{EVAL_CORPORA[lang3]}/default/{path.name}"
        mine = cache[cache_key(train, EVAL_CORPORA[lang3], BASELINE_LAMBDA)]["tokens"]
        if grid_key not in grid:
            print(f"!!! CROSS-CHECK MISSING: {grid_key} not in {GRID_CACHE_PATH}", flush=True)
            ok = False
        elif int(grid[grid_key]) != mine:
            print(f"!!! LAMBDA=0 MISMATCH {lang3}: mine={mine} grid={grid[grid_key]} ({grid_key})", flush=True)
            ok = False
        else:
            print(f"cross-check OK {lang3}: lambda=0 tokens {mine} == grid default", flush=True)
    return ok


def build_tables(cache: dict) -> tuple[list[str], list[list[str]], list[str], list[list[str]]]:
    def get(lang3: str, lam: float) -> dict:
        short = next(s for s, l3, _ in LANGUAGES if l3 == lang3)
        return cache[cache_key(train_corpus(short), EVAL_CORPORA[lang3], lam)]

    comp_header = ["lambda"] + [label for _, _, label in LANGUAGES] + ["Mean"]
    comp_rows = []
    for lam in LAMBDAS:
        deltas = []
        for _, lang3, _ in LANGUAGES:
            base = get(lang3, BASELINE_LAMBDA)["tokens"]
            deltas.append((get(lang3, lam)["tokens"] - base) / base * 100)
        comp_rows.append([f"{lam:g}"] + [f"{d:+.2f}" for d in deltas] + [f"{sum(deltas) / len(deltas):+.2f}"])

    morph_labels = [label for _, l3, label in LANGUAGES if l3 in MORPHALIGN_LANGS]
    morph_header = ["lambda"] + morph_labels + ["Geomean"]
    morph_rows = []
    for lam in LAMBDAS:
        vals = [morphalign_paper_score(get(l3, lam)["morphalign"]) for l3 in MORPHALIGN_LANGS]
        morph_rows.append([f"{lam:g}"] + [f"{v:.2f}" for v in vals] + [f"{geomean(vals):.2f}"])
    return comp_header, comp_rows, morph_header, morph_rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def tex_table(header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header).replace("lambda", r"$\lambda$") + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def write_tables(cache: dict) -> None:
    ch, cr, mh, mr = build_tables(cache)
    comp_caption = (
        r"Change in tokens on the Goldfish evaluation corpora (\%) relative to $\lambda=0$ "
        r"when $\lambda$ is subtracted from every token log-probability at inference."
    )
    morph_caption = (
        r"MorphAlign score ($\times 100$) of the same biased tokenizers; the last column is the geometric mean."
    )
    md = (
        "## Compression change vs lambda=0 (tokens, %; negative = fewer tokens)\n\n"
        + md_table(ch, cr)
        + "\n\n## MorphAlign x100\n\n"
        + md_table(mh, mr)
        + "\n"
    )
    tex = (
        tex_table(ch, cr, comp_caption, "tab:length-prior-compression")
        + "\n\n"
        + tex_table(mh, mr, morph_caption, "tab:length-prior-morphalign")
        + "\n"
    )
    md_path = paper_table_path(f"{TABLE_STEM}.md", extra=True)
    tex_path = paper_table_path(f"{TABLE_STEM}.tex", extra=True)
    md_path.write_text(md)
    tex_path.write_text(tex)
    print(md, flush=True)
    print(f"Saved: {md_path}\nSaved: {tex_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--dry-run", action="store_true", help="List tasks, check preconditions, exit.")
    args = parser.parse_args()

    tasks = build_tasks()
    check_preconditions()
    cache = load_cache(CACHE_PATH)
    if args.dry_run:
        for t in tasks:
            status = "cached" if t["cache_key"] in cache else "todo"
            print(f"{status}\t{t['cache_key']}\t{t['model_path']}")
        print("Preconditions OK (dry run).")
        return 0

    cache = compute(tasks, cache, args.max_workers)
    ok = crosscheck_baseline(cache)
    write_tables(cache)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
