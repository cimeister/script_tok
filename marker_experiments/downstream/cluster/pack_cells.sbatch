#!/usr/bin/env bash
# Several tokenizer cells concurrently on ONE node.
#
#     CELLS="de:plain de:bnd_w de:bnd_wp" sbatch .../pack_cells.sbatch
#
# Why concurrent and not one cell per node: on account a0229 every node carries 4 GPUs and
# a node-hour bills as 4 GPU-hours whether or not a GPU is touched. Tokenizer training is
# pure CPU, so one cell per node spends 4 GPU-hours an hour to run no GPU work.
#
# Packing only pays because a single cell cannot saturate 288 cores: corpus building
# alternates parallel encode bursts with a serial merge in the parent, which was measured
# at 91% of one core with the workers idle. Concurrent cells fill those gaps. If the work
# scaled perfectly with cores, packing would cost exactly the same node-hours and there
# would be no reason to do it.
#
# WORKERS is set per cell so the cells share the node rather than each claiming all of it,
# which is what train_multilang does by default (`workers: int = 0` means one per core).
#
#SBATCH --job-name=packcells
#SBATCH --account=a0229
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=results/multilang_grid/slurm/%x_%j.out
#SBATCH --error=results/multilang_grid/slurm/%x_%j.err

set -euo pipefail

REPO="${REPO:-/users/cmeister747/script_tok}"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh
# Pin imports to THIS checkout. The venv installs script_bpe and pynanochat as
# editable pointers into /users/cmeister747/script_tok, so a job launched from a
# worktree would otherwise run that tree library code, on whatever branch it happens
# to be on, with no record of it. Both entries are needed: REPO alone still resolves
# pynanochat to the main checkout. Harmless when REPO already is the main checkout.
export PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat:${PYTHONPATH:-}"

: "${CELLS:?set CELLS, space separated lang:arm pairs}"
# Required, not defaulted. A silent fallback to bpe here would run the wrong
# experiment: a mingram sweep would re-examine the bpe cells, find them present, and
# exit reporting success having trained nothing.
: "${TRAINERS:?set TRAINERS explicitly, bpe or mingram}"
NCELLS=$(wc -w <<< "${CELLS}")
CORES=$(nproc)
# Past this many workers a cell gets slower, not faster, so the share-out below is capped.
# Corpus building alternates parallel encode with a serial merge in the parent, and the
# parent's work grows with the number of partial counters, which is blocks times workers.
# More workers therefore shrink the parallel part and grow the serial one.
#
# Measured on the 21 cells of the FineWeb 5 GB grid, all building from the same cached
# sampled text: at 46 workers the median build was 2.66 h, at 56 workers 3.06 to 3.19 h,
# and one cell given the whole node at 280 workers had not finished after 7 h and was
# cancelled. The 46-worker pack beat the 56-worker packs while running six cells against
# five, so the ordering is not a contention artefact.
MAX_WORKERS_PER_CELL=48
WORKERS="${WORKERS:-$(( (CORES - 8) / NCELLS ))}"
if (( WORKERS > MAX_WORKERS_PER_CELL )); then
  echo "[pack] capping ${WORKERS} workers per cell at ${MAX_WORKERS_PER_CELL}; see the note above"
  WORKERS=${MAX_WORKERS_PER_CELL}
fi

echo "[pack] ${SLURM_JOB_ID:-nojob} node=$(hostname) cores=${CORES}"
echo "[pack] trainer=${TRAINERS}"
echo "[pack] ${NCELLS} cell(s) at ${WORKERS} workers each: ${CELLS}"
echo "[pack] commit=$(git rev-parse HEAD) repo=${REPO}"

pids=()
for cell in ${CELLS}; do
  lang="${cell%%:*}"; arm="${cell##*:}"
  log="results/multilang_grid/logs/${lang}_${arm}_${TRAINERS}.log"
  mkdir -p "$(dirname "$log")"
  ( uv run python -u marker_experiments/downstream/train_multilang.py \
        --lang "${lang}" --arms "${arm}" --trainers "${TRAINERS}" \
        --workers "${WORKERS}" --no-commit > "${log}" 2>&1 ) &
  pids+=($!)
  echo "[pack]   started ${lang}/${arm} -> ${log}"
done

# Report every failure rather than dying on the first, so one bad cell does not hide the
# rest of the node's work.
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[pack] cell $i FAILED"
    fail=1
  fi
done
echo "[pack] done, fail=${fail}"
exit "${fail}"
