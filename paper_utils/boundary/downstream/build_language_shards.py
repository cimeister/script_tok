"""Point a per-language nanochat base directory at FineWeb-2 text.

The downstream runner reads `shard_*.parquet` from `<NANOCHAT_BASE>/base_data_climbmix`,
and nanochat's loader takes every file but the last as training data and the last as
validation. That directory name is nanochat's, baked into its loader; for these runs it
holds FineWeb-2 in one language and no ClimbMix at all. `shard_provenance.json`, written
beside it, records what each shard really is, because the directory name does not.

Each shard is a symlink to a file in the shared read-only FineWeb-2 copy. Nothing is
duplicated, and the source files carry a `text` column, which is what the loader reads.

How many training shards a language gets is set by matching the number of passes over the
training text that the English runs made, about 3.5. Steps per run are fixed by the model,
so the tokens a run consumes are fixed too, and the passes over the text follow from how
much text there is and how many tokens the tokenizer makes of it. Korean text yields about
twice as many tokens per character as English, so the same number of passes needs less text.

    python -m paper_utils.boundary.downstream.build_language_shards \
        --lang ko --train-shards 3 --base-dir /capstor/scratch/cscs/$USER/marker_downstream/nanochat_base_ko
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import cyclopts

FINEWEB2 = Path(
    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots/data/output"
)
LANG_DIR = {"ko": "kor_Hang", "ru": "rus_Cyrl"}
# nanochat's own directory name, not a description of the contents. See the module docstring.
DATA_SUBDIR = "base_data_climbmix"

app = cyclopts.App()


@app.default
def build(
    lang: str,
    base_dir: Path,
    train_shards: int,
    source_dir: Path | None = None,
    measured_chars: Path | None = None,
):
    """Create `train_shards` training shards plus one validation shard, as symlinks.

    Args:
        lang: ko or ru.
        base_dir: the NANOCHAT_BASE for this language. Must not already hold shards.
        train_shards: how many training shards to link. One more file is linked for
            validation, which nanochat takes from the end of the sorted list.
        source_dir: override the FineWeb-2 language directory.
        measured_chars: JSON from the character count, recorded in the provenance file.
    """
    if lang not in LANG_DIR:
        raise SystemExit(f"unknown language {lang!r}; have {sorted(LANG_DIR)}")
    src = source_dir or (FINEWEB2 / LANG_DIR[lang])
    files = sorted(src.glob("*.parquet"))
    need = train_shards + 1
    if len(files) < need:
        raise SystemExit(f"{src} holds {len(files)} parquet files, fewer than the {need} needed")

    data_dir = base_dir / DATA_SUBDIR
    existing = sorted(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if existing:
        raise SystemExit(
            f"{data_dir} already holds {len(existing)} shard(s). Refusing to change the data "
            f"under a directory that runs may already have trained against. Remove it "
            f"explicitly if that is what you want."
        )
    data_dir.mkdir(parents=True, exist_ok=True)

    chars = {}
    if measured_chars is not None:
        chars = {e["file"]: e["chars"] for e in json.loads(measured_chars.read_text()).get(lang, [])}

    shards = []
    for i, source in enumerate(files[:need]):
        link = data_dir / f"shard_{i:05d}.parquet"
        link.symlink_to(source)
        shards.append(
            {
                "shard": link.name,
                "role": "validation" if i == need - 1 else "train",
                "source": str(source),
                "bytes": source.stat().st_size,
                "characters": chars.get(source.name),
            }
        )

    train_chars = sum(s["characters"] for s in shards if s["role"] == "train" and s["characters"])
    provenance = {
        "language": lang,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(src),
        "note": (
            f"{DATA_SUBDIR} is nanochat's directory name and holds no ClimbMix here: these "
            f"shards are FineWeb-2 {lang}. The last shard is validation, the rest training."
        ),
        "train_shards": train_shards,
        "train_characters": train_chars or None,
        "shards": shards,
    }
    (base_dir / "shard_provenance.json").write_text(json.dumps(provenance, indent=1))

    print(f"{lang}: {train_shards} training shards + 1 validation shard under {data_dir}")
    if train_chars:
        print(f"  training text: {train_chars:,} characters")
    print(f"  provenance: {base_dir / 'shard_provenance.json'}")


if __name__ == "__main__":
    app()
