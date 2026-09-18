#!/usr/bin/env bash
# Submit one language's downstream sweep: every arm x trainer x seed, four runs per node.
#
#     paper_utils/boundary/downstream/cluster/submit_language.sh --lang ko
#     paper_utils/boundary/downstream/cluster/submit_language.sh --lang ru --dry-run
#
# One job per (trainer, seed) holding that seed's four arms. Grouping this way means a
# finished job is a complete paired set for one seed, which is the unit the comparison
# comes in, rather than four seeds of one arm and nothing to pair them with.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

LANG_CODE=""
ARMS="plain,bnd_w,bnd_wpd,bnd_wpd_caps"
TRAINERS="bpe,mingram"
SEEDS="0,1,2"
ACCOUNT="a0229"
PARTITION="normal"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) LANG_CODE="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --trainers) TRAINERS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$LANG_CODE" ]] || { echo "set --lang" >&2; exit 1; }

# Training shards per language, chosen so each language makes about as many passes over its
# training text as the English runs did (about 3.5). Steps per run are fixed by the model, so
# the tokens a run consumes are fixed; passes follow from how much text there is and how many
# tokens the tokenizer makes of it. Korean yields roughly twice the tokens per character that
# Russian does, so it needs less text for the same number of passes. Recorded per language in
# shard_provenance.json beside the shards.
declare -A SHARDS=( [ko]=3 [ru]=7 )
NUM_SHARDS="${SHARDS[$LANG_CODE]:-}"
[[ -n "$NUM_SHARDS" ]] || { echo "no shard count recorded for language '$LANG_CODE'" >&2; exit 1; }

DATA_ROOT="/capstor/scratch/cscs/${USER}/marker_downstream"
BASE="${DATA_ROOT}/nanochat_base_${LANG_CODE}"
CORPUS="fineweb_${LANG_CODE}_5gb_quick"
OUT="results/marker_downstream_${LANG_CODE}"

[[ -d "$BASE" ]] || { echo "no base directory ${BASE}; run build_language_shards.py first" >&2; exit 1; }

# Untracked files count. `git diff` alone reports a clean tree while the file the job runs is
# untracked, which is how a sweep once ran code that was in no commit.
DIRTY=$(git status --porcelain | wc -l)
COMMIT=$(git rev-parse HEAD)
echo "repo ${REPO} at ${COMMIT}, ${DIRTY} uncommitted change(s)"
if (( DIRTY > 0 )); then
  echo "note: the jobs record this commit, and these files are not in it:"
  git status --porcelain | sed 's/^/  /'
fi
echo "language=${LANG_CODE} corpus=${CORPUS} shards=${NUM_SHARDS} base=${BASE}"
echo "arms=${ARMS} trainers=${TRAINERS} seeds=${SEEDS} account=${ACCOUNT} partition=${PARTITION}"

# sbatch refuses a job whose --output directory does not exist.
mkdir -p "${OUT}/slurm" "${OUT}/logs"

for trainer in ${TRAINERS//,/ }; do
  for seed in ${SEEDS//,/ }; do
    runs=""
    for arm in ${ARMS//,/ }; do
      tok="paper_utils/boundary/downstream/tokenizers/${CORPUS}_${arm}_${trainer}_v34685.json.gz"
      [[ -f "$tok" ]] || { echo "missing tokenizer ${tok}" >&2; exit 1; }
      runs+="${arm}:${seed} "
    done
    name="ds_${LANG_CODE}_${trainer}_s${seed}"
    echo "  ${name}: ${runs}"
    [[ "$DRY_RUN" == "1" ]] && continue
    jid=$(sbatch --parsable \
      --account="${ACCOUNT}" --partition="${PARTITION}" \
      --job-name="${name}" \
      --output="${OUT}/slurm/${name}_%j.out" --error="${OUT}/slurm/${name}_%j.err" \
      --export=ALL,REPO="${REPO}",RUNS="${runs}",TRAINER="${trainer}",CORPUS="${CORPUS}",NUM_SHARDS="${NUM_SHARDS}",OUT="${OUT}",NANOCHAT_BASE_REQUESTED="${BASE}" \
      paper_utils/boundary/downstream/cluster/pack_runs.sbatch)
    echo "    submitted ${jid}"
  done
done
