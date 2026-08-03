#!/usr/bin/env bash
# The bnd_wpd_extcaps downstream sweep: one arm, seeds 0-2, matched to the BPE runs.
#
#     CORE_SAFE_ARMS="" marker_experiments/downstream/cluster/submit_extcaps.sh
#
# bnd_wpd_extcaps differs from bnd_wpd_caps only in which side of the boundary markers the
# caps code sits on: `The` is `<^><|>the<|>` here against `<|><^>the<|>` there. The
# lowercase span is therefore a suffix of the cased one and BPE may cover both with one
# piece. Every other setting matches the committed BPE runs so the difference in the
# downstream numbers is attributable to that one change.
#
# ONE job for all three seeds, not three. A node carries 4 GPUs and a node-hour bills as 4
# GPU-hours whether or not a GPU is touched, so three single-GPU runs on three nodes bill
# 12 GPU-hours an hour to use 3. They run concurrently on GPUs 0, 1 and 2 of one node.
# Concurrency is safe because runner.py gives each run its own base dir keyed by
# tokenizer_id, which carries the seed, and symlinks the shared read-only ClimbMix shards
# and CORE bundle into it; the bpb byte table that concurrent runs used to race on is
# per-run. ENCODE_WORKERS is set because run_downstream_eval otherwise sizes its encode
# pool for a run that owns the whole node.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh

ARM=${ARM:-bnd_wpd_extcaps}
# Which arms CORE can score. Required, and MAY be empty: run core_prefix_check.py on the
# trained tokenizer and pass what it supports. Getting it wrong loses the whole run:
# base_eval raises on the first violation and reports nothing, bits-per-byte included.
# No `:` in the test, so an explicit empty value is honoured.
: "${CORE_SAFE_ARMS?set CORE_SAFE_ARMS explicitly; the empty string is a valid value}"
SEEDS=${SEEDS:-0,1,2}
TRAINER=${TRAINER:-bpe}
DEPTH=${DEPTH:-12}
VOCAB=${VOCAB:-34685}
CORPUS=${CORPUS:-fineweb_en_5gb}
NUM_SHARDS=${NUM_SHARDS:-8}
GPUS=${GPUS:-1}
TRAIN_WORKERS=${TRAIN_WORKERS:-90}
ENCODE_WORKERS=${ENCODE_WORKERS:-90}
# Not `${OUT:-...}`: env.sh has already exported OUT=results/marker_downstream, so
# defaulting from OUT would drop this sweep's logs in with the committed BPE runs and
# collect_results.py would then rebuild one TSV over both.
OUT_DIR=${OUT_DIR:-results/extcaps_downstream}
NANOCHAT_BASE_DIR=/capstor/scratch/cscs/cmeister747/marker_downstream/nanochat_base

if ! git diff --quiet HEAD || ! git diff --cached --quiet HEAD; then
  echo "refusing to submit: working tree differs from HEAD, so the jobs would run code" >&2
  echo "no commit describes." >&2
  git status --porcelain >&2
  exit 1
fi

TOK="marker_experiments/downstream/tokenizers/${CORPUS}_${ARM}_${TRAINER}_v${VOCAB}.json.gz"
[[ -f "$TOK" ]] || { echo "missing tokenizer: $TOK" >&2; exit 1; }

# Verify the tokenizer is exactly the matched size before spending any GPU time. The
# adapter reports vocab+1 for the synthetic BOS.
n=$(PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat" uv run python -c "
from marker_experiments.downstream.boundary_tokenizer import BoundaryBPETokenizer
from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
print(ScriptBPETokenizerAdapter(BoundaryBPETokenizer.load('$TOK')).get_vocab_size())
")
if [[ "$n" != "$((VOCAB + 1))" ]]; then
  echo "refusing to submit: ${ARM} vocabulary is ${n}, expected $((VOCAB + 1))" >&2
  exit 1
fi
echo "-- ${ARM}: vocabulary ${n}, matched"

# The shards must already be there. runner.py refuses to download from inside a run, so a
# missing shard would fail three runs an hour in rather than here.
have=$(ls "${NANOCHAT_BASE_DIR}/base_data_climbmix"/shard_*.parquet 2>/dev/null | wc -l)
if (( have < NUM_SHARDS + 1 )); then
  echo "refusing to submit: ${NANOCHAT_BASE_DIR}/base_data_climbmix holds ${have} shard(s)," >&2
  echo "and this sweep declares ${NUM_SHARDS} train shards plus a val shard." >&2
  exit 1
fi
echo "-- shards: ${have} present, ${NUM_SHARDS} train + 1 val required"

mkdir -p "${OUT_DIR}/slurm" "${OUT_DIR}/logs"
IN_FLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
JOB="extcaps_${ARM}_${TRAINER}_d${DEPTH}"
if grep -qx "${JOB}" <<< "$IN_FLIGHT"; then
  echo "-- ${JOB}: already queued or running, not submitting"; exit 0
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
TODO=()
for seed in "${SEED_LIST[@]}"; do
  log="${OUT_DIR}/logs/${ARM}_${TRAINER}_d${DEPTH}_s${seed}.log"
  if [[ -s "$log" ]] && grep -q "artifact_dir" "$log"; then
    echo "-- seed ${seed}: already has a result, not submitting"; continue
  fi
  TODO+=("$seed")
done
if (( ${#TODO[@]} == 0 )); then echo "nothing to do"; exit 0; fi
# One GPU per seed, one node, so the node's 4 GPUs are the ceiling. Past that the loop
# below would hand two seeds the same CUDA_VISIBLE_DEVICES and they would share a GPU.
GPUS_PER_NODE=4
if (( ${#TODO[@]} > GPUS_PER_NODE )); then
  echo "refusing to submit: ${#TODO[@]} seeds but only ${GPUS_PER_NODE} GPUs on a node." >&2
  echo "Run them in batches, or extend this script to allocate more nodes." >&2
  exit 1
fi
echo "-- submitting seeds: ${TODO[*]}"

# Build the concurrent launch block: one run_arms.sh per seed, each pinned to its own GPU.
LAUNCH=""
gpu=0
for seed in "${TODO[@]}"; do
  LAUNCH+="
CUDA_VISIBLE_DEVICES=${gpu} ARMS=${ARM} SEEDS=${seed} \\
  marker_experiments/downstream/run_arms.sh > \"${OUT_DIR}/slurm/\${SLURM_JOB_ID}_s${seed}.out\" 2>&1 &
pids+=(\$!); seeds+=(${seed})
echo \"[pack] seed ${seed} on GPU ${gpu}, pid \${pids[-1]}\"
"
  gpu=$((gpu + 1))
done

jid=$(sbatch --parsable --job-name="${JOB}" --account=infra01 --partition=normal \
    --nodes=1 --ntasks-per-node=1 --time=12:00:00 \
    --output="${OUT_DIR}/slurm/%x_%j.out" --error="${OUT_DIR}/slurm/%x_%j.err" \
    --wrap="
# No -e: a failing seed must reach the wait loop below and be reported, not abort the job.
set -uo pipefail
cd ${REPO} || exit 1
source marker_experiments/downstream/cluster/env.sh
# The venv installs script_bpe and pynanochat as editable pointers into the MAIN
# checkout, so without this the job runs that tree's library code no matter which
# checkout the driver script came from, and the clean-tree guard checks the wrong
# tree. Both entries are needed: REPO alone still resolves pynanochat to the main one.
export PYTHONPATH=\"${REPO}:${REPO}/eval/py-nanochat:\${PYTHONPATH:-}\"
export NANOCHAT_BASE=${NANOCHAT_BASE_DIR}
export OUT=${OUT_DIR}
export TRAINER=${TRAINER} DEPTH=${DEPTH} GPUS=${GPUS}
export CORE_SAFE_ARMS=\"${CORE_SAFE_ARMS}\"
# run_arms.sh ends by copying \$OUT/results.tsv over marker_experiments/paper/generated
# and regenerating the tables. This sweep TSV holds three extcaps rows, and those paper
# artifacts are tracked and carry the full BPE grid. Keep them out of it.
export SKIP_PAPER_ARTIFACTS=1
export VOCAB=${VOCAB} CORPUS=${CORPUS} NUM_SHARDS=${NUM_SHARDS}
export TRAIN_WORKERS=${TRAIN_WORKERS} ENCODE_WORKERS=${ENCODE_WORKERS}
COMMIT=\$(git rev-parse HEAD)
MAIN_COMMIT=\$(git -C /users/cmeister747/script_tok rev-parse HEAD)
MAIN_DIRTY=\$(git -C /users/cmeister747/script_tok status --porcelain | wc -l)
VENDOR=\$(git -C eval/py-nanochat/vendor/nanochat rev-parse HEAD)
TOK_SHA=\$(sha256sum ${TOK} | cut -d' ' -f1)
echo \"[job] \${SLURM_JOB_ID} ${JOB} commit=\${COMMIT}\"
echo \"[job] main_checkout=\${MAIN_COMMIT} dirty=\${MAIN_DIRTY} nanochat=\${VENDOR}\"
echo \"[job] tokenizer=${TOK}\"
echo \"[job] tokenizer_sha256=\${TOK_SHA}\"
echo \"[job] node=\$(hostname) cores=\$(nproc) gpus=\$(nvidia-smi -L | wc -l)\"
echo \"[job] seeds=${TODO[*]} core_safe_arms='${CORE_SAFE_ARMS}' encode_workers=${ENCODE_WORKERS}\"
uv pip freeze > \"${OUT_DIR}/slurm/\${SLURM_JOB_ID}.freeze\" 2>/dev/null || true

pids=(); seeds=()
${LAUNCH}
# Report every failure rather than dying on the first, so one bad seed does not hide the
# other two.
fail=0
for i in \"\${!pids[@]}\"; do
  if wait \"\${pids[\$i]}\"; then
    echo \"[pack] seed \${seeds[\$i]} ok\"
  else
    echo \"[pack] seed \${seeds[\$i]} FAILED\"
    fail=1
  fi
done

# One authoritative collection after the join. Each run_arms.sh also collects when it
# finishes, and three concurrent writers to one TSV can interleave; this pass runs when
# every log is complete and overwrites whatever they left.
uv run python marker_experiments/downstream/collect_results.py \\
    --logs-dir \"${OUT_DIR}/logs\" --out \"${OUT_DIR}/results.tsv\" || fail=1
echo \"[pack] done, fail=\${fail}\"
exit \${fail}
")
echo "-- ${JOB}: job ${jid}, ${#TODO[@]} seed(s) on one node"
