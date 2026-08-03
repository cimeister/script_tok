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
VOCAB=${VOCAB:-34685}
CORPUS=${CORPUS:-fineweb_en_5gb}
OUT_DIR=results/mingram_downstream

if ! git diff --quiet HEAD || ! git diff --cached --quiet HEAD; then
  echo "refusing to submit: working tree differs from HEAD, so the jobs would run code" >&2
  echo "no commit describes." >&2
  git status --porcelain >&2
  exit 1
fi

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
    tag="${arm}_${TRAINER}_d12_s${seed}"
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
export NANOCHAT_BASE=/capstor/scratch/cscs/cmeister747/marker_downstream/nanochat_base
export OUT=${OUT_DIR}
export ARMS=${arm} SEEDS=${seed} TRAINER=${TRAINER} DEPTH=12 GPUS=1
export CORE_SAFE_ARMS=\"${CORE_SAFE_ARMS}\"
export VOCAB=${VOCAB} CORPUS=${CORPUS} NUM_SHARDS=8 TRAIN_WORKERS=90
echo \"[job] \${SLURM_JOB_ID} ${tag} commit=\$(git rev-parse HEAD) nanochat=\$(git -C eval/py-nanochat/vendor/nanochat rev-parse HEAD)\"
srun -ul marker_experiments/downstream/run_arms.sh
")
    echo "-- ${tag}: job ${jid}"
  done
done
