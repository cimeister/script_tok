"""Restore original quick samples, verify saved content, then resume the large run."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", str(ROOT / "results/boundary_restore_hf"))
    from script_bpe.corpus.bilingual import bounded_documents, source_cache
    from script_bpe.corpus.registry import create_streaming_quick_corpus, normalize_whitespace
    from script_bpe.utils import create_logger

    base = ROOT / "results/corpora"
    manifest_path = base / "_bilingual_text/fineweb_enko_2000m_en50_local_v2/manifest.json"
    saved = json.loads(manifest_path.read_text())
    refreshed = json.loads(json.dumps(saved))
    for lang, info in saved["sources"].items():
        dataset = "HuggingFaceFW/fineweb" if lang == "en" else "HuggingFaceFW/fineweb-2"
        config = "sample-10BT" if lang == "en" else "kor_Hang"
        print(f"Restoring original 5G quick sample: {lang}", flush=True)
        create_streaming_quick_corpus(
            dataset_name=dataset, corpus_name=f"fineweb_{lang}_5gb_quick", pretokenizer=None,
            base_dir=str(base), logger=create_logger("restore"), sample_max_chars=5_000_000_000,
            text_transform=normalize_whitespace, name=config, split="train", text_only=True,
        )
        source, paths = source_cache(base, lang)
        if hashlib.sha256(source.read_bytes()).hexdigest() != info["manifest_sha256"]:
            raise RuntimeError(f"{lang}: restored source manifest differs; checkpoint will NOT be resumed")
        digest = hashlib.sha256()
        for text in bounded_documents(paths, saved["budgets"][lang]):
            digest.update((json.dumps(text, ensure_ascii=False) + "\n").encode("utf-8"))
        if digest.hexdigest() != info["sample_sha256"]:
            raise RuntimeError(f"{lang}: training text differs; checkpoint will NOT be resumed")
        old_sizes = [(row[0], row[1]) for row in info["block_stats"]]
        if [(p.name, p.stat().st_size) for p in paths] != old_sizes:
            raise RuntimeError(f"{lang}: restored block sizes differ; checkpoint will NOT be resumed")
        refreshed["sources"][lang]["block_stats"] = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        print(f"Verified identical {lang} training sample", flush=True)
    backup = manifest_path.with_name("manifest.before_restore.json")
    if not backup.exists():
        backup.write_text(manifest_path.read_text())
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(refreshed, indent=2) + "\n")
    temporary.replace(manifest_path)
    print("Both samples verified; resuming checkpointed training", flush=True)
    subprocess.run([sys.executable, "-m", "paper_utils.boundary.run_bilingual",
                    "--million-chars", "2000", "--workers", "4", "--max-rss-gib", "20"], check=True)
    subprocess.run([sys.executable, "-m", "paper_utils.boundary.summarize_bilingual"], check=True)


if __name__ == "__main__":
    main()
