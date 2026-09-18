# Design choices: Korean and Russian downstream runs

Every choice made in setting up the Korean and Russian language-model comparison, started
2026-09-17. The English runs this extends are in `paper/generated/results*.tsv`.

## What is being run

Per language, 4 schemes x 2 trainers x 3 seeds = 24 runs, and 48 in total.

| | |
|---|---|
| languages | Korean, Russian |
| schemes | `plain`, `bnd_w`, `bnd_wpd`, `bnd_wpd_caps` |
| trainers | BPE, MinGram |
| seeds | 0, 1, 2 |
| model | nanochat, 12 layers, one GPU per run, as in the English runs |
| metric | validation bits per true byte |

## Tokenizers

The 16 tokenizers are the `_quick` cells of the six-language grid, at total vocabulary
34,685, the same files the published intrinsic tables were computed from. They are
gitignored upstream and were restored from `fork/claude/mingram-five-arms`, where the files
the grid produced are committed to git.

Two things about the restore are worth stating plainly.

**The caps arm is the file the grid calls `bnd_wpd_extcapsfix`.** Upstream renamed that arm
to `bnd_wpd_caps` when the work moved to `main`, and dropped the earlier arm of the same
name, which placed the case code inside the span's markers. The 12 intrinsic cells upstream
reports as `bnd_wpd_caps` match the `extcapsfix` cells of the grid on training time and
vocabulary sizes. The restored file is named for the upstream arm.

**The stored configuration was relabelled, and the relabelling was verified rather than
assumed.** The August files serialize `boundary_targets` with a `word` entry, a `caps_codes`
flag and, for the caps arm, a class name that current code does not define, so current code
refuses to load them. `migrate_legacy_tokenizer_config.py` rewrites those few fields and
keeps the rewrite only if the August code reading the original file and current code reading
the rewritten file produce identical token ids over 200 documents of the language. All 12
passed, with the vocabulary unchanged at 34,685. Before that, the August and current
pretokenizers were compared directly: identical atomic tokens in identical order, and
identical output on 800 Korean and Russian documents.

## Training data

FineWeb-2, one language per model, from the copy at

    /capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots

Not the ClimbMix data the English runs used, which is English. The files carry a `text`
column, which is what nanochat's loader reads, so they are linked in rather than converted.

`build_language_shards.py` links each language's shards into its own base directory and
writes `shard_provenance.json` beside them. The directory nanochat reads is called
`base_data_climbmix` and holds no ClimbMix here; that name is nanochat's, and the provenance
file is what records the truth.

**How much text each language gets.** Steps per run are fixed by the model, so the tokens a
run consumes are fixed, and the number of passes over the training text follows from how much
text there is and how many tokens the tokenizer makes of it. The English runs passed over
about 2e9 characters about 3.5 times. Matching that:

| | characters per token, plain | training characters | shards |
|---|---|---|---|
| English (reference) | 4.46 | 2.0e9 | 8 |
| Korean | 2.30 | 1.22e9 | 3 |
| Russian | 4.32 | 2.05e9 | 7 |

Characters per token are the measured values from `eval_goldfish.json`, so the pass counts
are approximate; the exact figure per run is in its log. Matching the number of passes, rather
than the number of characters, means each language repeats its training text about as often as
the English runs repeated theirs. Matching characters instead would have given Korean about
1.5 passes, which is a different setting from every other number in the paper. The 32-shard
appendix run, a single pass, reproduced the English result, so this choice probably does not
change the outcome. It is still a choice.

The comparison that carries the claim is between schemes within one language, and all schemes
of a language train on identical text, so the text budget does not enter it.

## No CORE

CORE is an English benchmark and is not scored. Current `run_arms.sh` scores no arm on CORE
in any case: the arms that mark punctuation break the prompt-prefix property its
language-modeling tasks assert, and scoring only the surviving arms would compare a subset.

## Scheduling

Account `a0229`, partition `normal`, four runs per node with one GPU each, submitted as one
job per (trainer, seed) holding that seed's four arms. A node bills 4 GPU-hours per hour
whether or not the GPUs are used, and a single run leaves most of the node idle, so one run
per node would cost four times as much. A finished job is a complete paired set
for one seed, which is the unit the comparison comes in.

Estimated cost, from the August measurement of 4 runs finishing in 2h01m: about 48 GPU-hours
per language.

## Two differences from the English runs, recorded because they are not visible in the numbers

- `run_arms.sh` gained an `ENCODE_WORKERS` passthrough. Without it each run sizes its
  tokenization pool for a whole node, and four co-resident runs would ask for about 1,150
  worker processes on 288 cores. The August launcher had this; the squash to `main` dropped it.
- `env.sh` no longer overwrites a caller's `NANOCHAT_BASE`. Each language has its own base
  directory holding its own text, and the August version would have silently pointed every
  language at one of them.

## Risks specific to this setup

- The training text and the tokenizers are chosen independently, so Korean tokenizers could
  be pointed at Russian text and would produce complete, plausible numbers. The launcher
  refuses to start unless the language recorded beside the shards matches the corpus name in
  the tokenizer file names.
- capstor deletes files not accessed for 14 days, which is what destroyed this project's
  environment, data shards, checkpoints and logs between 2026-08-04 and 2026-09-17. Anything
  worth keeping past a sweep should be copied off scratch.
