#!/usr/bin/env bash
# The MinGram downstream sweep: plain and bnd_wpd, seeds 0-2, matched to the BPE runs.
#
#     marker_experiments/downstream/cluster/submit_mingram.sh
#
# Two arms rather than four. The question is whether the downstream conclusion is specific
# to greedy merge training, and plain against bnd_wpd is the comparison that carries it:
# bnd_wpd is the arm whose gain cannot come from spending more forward passes per byte,
# because it emits fewer tokens per byte than plain, not more.
#
# TAG_SUFFIX keeps these off the BPE runs' logs and checkpoint directories. The tag already
# carries the trainer, so this is belt and braces, but the two sweeps also differ in
# NANOCHAT_BASE and OUT and it should be obvious which is which.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh

ARMS=${ARMS:-plain,bnd_wpd}
# Which arms CORE can score, measured on the MinGram tokenizers by mingram_preflight.py
# rather than inherited from run_arms.sh's default. That default was measured on BPE
# tokenizers, and MinGram segments with a dynamic program instead of greedy merge replay,
# so an arm CORE-safe under BPE need not be under MinGram. Getting it wrong loses the whole
# run: base_eval raises on the first violation and reports nothing, bits-per-byte included.
: "${CORE_SAFE_ARMS:?run mingram_preflight.py and pass the CORE_SAFE_ARMS it prints}"
SEEDS=${SEEDS:-0,1,2}
TRAINER=mingram
DEPTH=${DEPTH:-12}
VOCAB=${VOCAB:-34685}
CORPUS=${CORPUS:-fineweb_en_5gb}
OUT_DIR=results/mingram_downstream

if ! git diff --quiet HEAD || ! git diff --cached --quiet HEAD; then
  echo "refusing to submit: working tree differs from HEAD, so the jobs would run code" >&2
  echo "no commit describes." >&2
  git status --porcelain >&2
  exit 1
fi

# Verify each tokenizer is exactly the matched size before spending any GPU time. MinGram
# prunes down to a target and can land short; the adapter reports vocab+1 for the synthetic
# BOS. This is the one check that must not depend on an operator remembering to run the
# preflight and read its output.
for arm in ${ARMS//,/ }; do
  tok="marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
  [[ -f "$tok" ]] || { echo "missing tokenizer: $tok" >&2; exit 1; }
  n=$(PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat" uv run python -c "
import sys
from marker_experiments.downstream.boundary_tokenizer import BoundaryMinGramModel
from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
print(ScriptBPETokenizerAdapter(BoundaryMinGramModel.load('$tok')).get_vocab_size())
")
  if [[ "$n" != "$((VOCAB + 1))" ]]; then
    echo "refusing to submit: ${arm} vocabulary is ${n}, expected $((VOCAB + 1))" >&2
    exit 1
  fi
  echo "-- ${arm}: vocabulary ${n}, matched"
done

mkdir -p "${OUT_DIR}/slurm" "${OUT_DIR}/logs"
IN_FLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

IFS=',' read -ra ARM_LIST <<< "$ARMS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

# Seed-major: a partially drained queue then leaves a full two-arm comparison at one seed,
# which is usable, rather than three seeds of one arm, which answers nothing.
for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARM_LIST[@]}"; do
    tok="marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
    if [[ ! -f "$tok" ]]; then
      echo "-- ${arm}/s${seed}: no tokenizer at ${tok}, skipping" >&2
      continue
    fi
    tag="${arm}_${TRAINER}_d${DEPTH}_s${seed}"
    log="${OUT_DIR}/logs/${tag}.log"
    if [[ -s "$log" ]] && grep -q "artifact_dir" "$log"; then
      echo "-- ${tag}: already has a result, not submitting"; continue
    fi
    if grep -qx "mgds_${tag}" <<< "$IN_FLIGHT"; then
      echo "-- ${tag}: already queued or running, not submitting"; continue
    fi
    jid=$(sbatch --parsable --job-name="mgds_${tag}" --account=a0229 --partition=normal \
        --nodes=1 --ntasks-per-node=1 --time=12:00:00 \
        --output="${OUT_DIR}/slurm/%x_%j.out" --error="${OUT_DIR}/slurm/%x_%j.err" \
        --wrap="
set -euo pipefail
cd ${REPO}
source marker_experiments/downstream/cluster/env.sh
# The venv installs script_bpe and pynanochat as editable pointers into the MAIN
# checkout, so without this the job runs that tree library code no matter which
# checkout the driver script came from, and the clean-tree guard checks the wrong
# tree. Both entries are needed: REPO alone still resolves pynanochat to the main one.
export PYTHONPATH=\"${REPO}:${REPO}/eval/py-nanochat:\${PYTHONPATH:-}\"
export NANOCHAT_BASE=/capstor/scratch/cscs/cmeister747/marker_downstream/nanochat_base
export OUT=${OUT_DIR}
export ARMS=${arm} SEEDS=${seed} TRAINER=${TRAINER} DEPTH=${DEPTH} GPUS=1
export CORE_SAFE_ARMS=\"${CORE_SAFE_ARMS}\"
# run_arms.sh ends by copying $OUT/results.tsv over marker_experiments/paper/generated/
# and regenerating the tables. This sweep TSV holds only mingram rows, and those paper
# artifacts are tracked and carry the BPE numbers. Keep them out of it.
export SKIP_PAPER_ARTIFACTS=1
export VOCAB=${VOCAB} CORPUS=${CORPUS} NUM_SHARDS=8 TRAIN_WORKERS=90
COMMIT=\$(git rev-parse HEAD)
MAIN_COMMIT=\$(git -C /users/cmeister747/script_tok rev-parse HEAD)
MAIN_DIRTY=\$(git -C /users/cmeister747/script_tok status --porcelain | wc -l)
VENDOR=\$(git -C eval/py-nanochat/vendor/nanochat rev-parse HEAD)
TOK_SHA=\$(sha256sum marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz | cut -d" " -f1)
echo \"[job] \${SLURM_JOB_ID} ${tag} commit=\${COMMIT}\"
echo \"[job] main_checkout=\${MAIN_COMMIT} dirty=\${MAIN_DIRTY} nanochat=\${VENDOR}\"
echo \"[job] tokenizer_sha256=\${TOK_SHA}\"
uv pip freeze > \"${OUT_DIR}/slurm/\${SLURM_JOB_ID}.freeze\" 2>/dev/null || true
srun -ul marker_experiments/downstream/run_arms.sh
")
    echo "-- ${tag}: job ${jid}"
  done
done
