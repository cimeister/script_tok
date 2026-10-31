import pytest

from paper_utils.hybrid.train_hybrid import bpe_to_unigram_tokens
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig

PRETOKENIZER_NAME = "bytes_gpt4"
# Chosen so that the tiny Taylor Swift corpus needs several prune rounds (checked via metadata["num_iterations"]).
CONFIG_KWARGS = dict(additional_vocab_size=20, initial_vocab_factor=150, pruning_shrinking_factor=0.5, verbose=False)
INTERRUPT_AFTER_CHECKPOINTS = 2
BPE_INIT_SIZE = 1200  # BPE tokens added to the forced initial vocabulary


class Interrupted(Exception):
    pass


@pytest.fixture(scope="module")
def swift_corpus(tmp_path_factory):
    with open("tests/data/taylorswift.txt", "r") as f:
        text = f.read()
    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = PretokenizedCorpus.from_texts(
        "test_unigram_checkpoint",
        texts=[text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path_factory.mktemp("corpus")),
    )
    return pretokenizer, corpus


def make_trainer(swift_corpus, checkpoint_path=None, **overrides):
    pretokenizer, corpus = swift_corpus
    cfg = UnigramTrainerConfig(**{**CONFIG_KWARGS, "checkpoint_path": checkpoint_path, **overrides})
    return UnigramTrainer(pretokenizer, corpus, cfg)


def interrupt_after_checkpoints(monkeypatch, n):
    """Simulate a job killed right after the n-th checkpoint was written."""
    original = UnigramTrainer.write_checkpoint
    calls = []

    def write_then_die(self, *args, **kwargs):
        original(self, *args, **kwargs)
        calls.append(1)
        if len(calls) == n:
            raise Interrupted

    monkeypatch.setattr(UnigramTrainer, "write_checkpoint", write_then_die)
    return lambda: monkeypatch.setattr(UnigramTrainer, "write_checkpoint", original)


def small_bpe_initial_vocab(swift_corpus):
    """Fresh forced initial vocabulary from a BPE model, as train_model.py builds it for bpe_init methods.

    Built anew for every run because the unigram M-step mutates the log_probs of the tokens it is given."""
    pretokenizer, corpus = swift_corpus
    bpe = BPETrainer(pretokenizer, corpus, BPETrainerConfig(additional_vocab_size=BPE_INIT_SIZE, num_workers=1)).train()
    return bpe_to_unigram_tokens(bpe)


# Each variant must need more prune rounds than INTERRUPT_AFTER_CHECKPOINTS (asserted in the test).
VARIANTS = {
    "default": dict(),
    "fsp": dict(flat_score_prune=True, pre_final_vocab_factor=1.0),
    "bpe_init": dict(forced=True),
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_resume_after_interruption_equals_uninterrupted_run(swift_corpus, tmp_path, monkeypatch, variant):
    """A run killed after prune round 2 and resumed from its checkpoint must end in exactly the uninterrupted model,
    for the default, flat-score-prune and forced (BPE) initial vocabulary paths."""
    overrides = dict(VARIANTS[variant])
    forced = overrides.pop("forced", False)

    def trainer(checkpoint_path=None):
        if forced:
            overrides["forced_initial_vocab"] = small_bpe_initial_vocab(swift_corpus)
        return make_trainer(swift_corpus, checkpoint_path, **overrides)

    ckpt = str(tmp_path / "model.ckpt")
    model_a = trainer().train()
    assert model_a.metadata["num_iterations"] > INTERRUPT_AFTER_CHECKPOINTS, "test needs more rounds than it interrupts"

    flat_prune_calls = []
    original_flat_prune = UnigramTrainer.score_based_prune
    monkeypatch.setattr(
        UnigramTrainer,
        "score_based_prune",
        lambda self, *args, **kwargs: flat_prune_calls.append(1) or original_flat_prune(self, *args, **kwargs),
    )
    restore = interrupt_after_checkpoints(monkeypatch, INTERRUPT_AFTER_CHECKPOINTS)
    with pytest.raises(Interrupted):
        trainer(ckpt).train()
    restore()
    # Flat-score pruning is used in every prune round of the fsp variant and never by the others before finalize.
    assert len(flat_prune_calls) == (INTERRUPT_AFTER_CHECKPOINTS if variant == "fsp" else 0)

    model_b = trainer(ckpt).train()

    assert model_b.metadata["num_iterations"] == model_a.metadata["num_iterations"]
    assert {t.atomic_tokens.tobytes() for t in model_b.tokens.values()} == {
        t.atomic_tokens.tobytes() for t in model_a.tokens.values()
    }
    # Both runs do identical floating point operations in identical order on JSON round-tripped (exact) floats.
    state_a = {t.atomic_tokens.tobytes(): (t.log_prob, t.required) for t in model_a.tokens.values()}
    state_b = {t.atomic_tokens.tobytes(): (t.log_prob, t.required) for t in model_b.tokens.values()}
    assert state_a == state_b
    assert [t.id for t in model_a.tokens.values()] == [t.id for t in model_b.tokens.values()]
    for key in ("objective", "total_tokens", "totals_removed", "num_defended", "initial_vocab_size"):
        assert model_b.metadata[key] == model_a.metadata[key], key
    assert model_b.metadata["config"] == model_a.metadata["config"]


def test_checkpoint_with_mismatched_config_raises(swift_corpus, tmp_path):
    """A checkpoint from a different configuration must never be silently resumed or overwritten."""
    ckpt = str(tmp_path / "model.ckpt")
    make_trainer(swift_corpus, ckpt).train()
    mismatched = make_trainer(swift_corpus, ckpt, additional_vocab_size=CONFIG_KWARGS["additional_vocab_size"] + 1)
    with pytest.raises(ValueError, match="additional_vocab_size"):
        mismatched.train()


def test_checkpoint_from_different_corpus_raises(swift_corpus, tmp_path):
    """A checkpoint written for another corpus (different name/metadata) must be rejected."""
    ckpt = str(tmp_path / "model.ckpt")
    make_trainer(swift_corpus, ckpt).train()
    pretokenizer, _ = swift_corpus
    other = PretokenizedCorpus.from_texts(
        "other_corpus", texts=["a short different text"], pretokenizer=pretokenizer, base_path=str(tmp_path)
    )
    trainer = UnigramTrainer(pretokenizer, other, UnigramTrainerConfig(**CONFIG_KWARGS, checkpoint_path=ckpt))
    with pytest.raises(ValueError, match="corpus_name"):
        trainer.train()
