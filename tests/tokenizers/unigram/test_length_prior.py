"""Tests for the token-count (length) prior variants of the Unigram trainer.

The standard Unigram LM drops the prior P(M=m) over the number of tokens per pretoken. "eow" restores it as a
geometric prior with learned parameter (end-of-word outcome in the categorical); "geometric" restores it as a fixed
per-token penalty lambda (= token_bias) that is applied in training, pruning, and at inference.
"""

import copy
import json
import math

import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.utils import token_array

PRETOKENIZER_NAME = "scriptenc_cb"
# Hand-built models need one distinct atomic id per character; script encodings share a leading script id.
BYTES_PRETOKENIZER_NAME = "bytes_gpt4_cb"
ADDITIONAL_VOCAB = 60
GEO_LAMBDA = 2.0
HUGE_LAMBDA = 10.0
NORMALIZATION_TOL = 1e-9


@pytest.fixture(scope="module")
def pretokenizer():
    return get_pretokenizer(PRETOKENIZER_NAME)


@pytest.fixture(scope="module")
def swift_text():
    with open("tests/data/taylorswift.txt", "r") as f:
        return f.read()


@pytest.fixture(scope="module")
def swift_corpus(pretokenizer, swift_text, tmp_path_factory):
    return PretokenizedCorpus.from_texts(
        "length_prior_swift",
        texts=[swift_text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path_factory.mktemp("lp_corpus")),
    )


@pytest.fixture(scope="module")
def train(pretokenizer, swift_corpus):
    cache: dict = {}

    def _train(**kwargs) -> UnigramModel:
        key = json.dumps(kwargs, sort_keys=True)
        if key not in cache:
            cfg = UnigramTrainerConfig(additional_vocab_size=ADDITIONAL_VOCAB, **kwargs)
            cache[key] = UnigramTrainer(pretokenizer, swift_corpus, cfg).train()
        return cache[key]

    return _train


def _shifted_copy(model: UnigramModel, shift: float) -> UnigramModel:
    """A prior-free model whose log_probs are those of `model` shifted by `shift`."""
    tokens = [
        UnigramToken(id=t.id, atomic_tokens=t.atomic_tokens, log_prob=t.log_prob + shift, required=t.required)
        for t in model.tokens.values()
    ]
    return UnigramModel(model.pretokenizer, tokens)


# ---- 1. backward compatibility of the data format ----


def test_no_prior_model_has_no_length_prior_and_unchanged_file_format(train, tmp_path):
    """A model trained without a prior must serialize exactly like the legacy format (no new key), and a legacy
    file lacking the field must load with length_prior None and no inference bias."""
    model = train()
    assert model.length_prior is None
    assert model.metadata["config"]["length_prior"] == "none"
    assert model.metadata["config"]["token_bias"] == 0.0

    path = model.save(str(tmp_path / "m.json"))
    with open(path) as f:
        data = json.load(f)
    assert "length_prior" not in data

    legacy_path = tmp_path / "legacy.json"  # old format: dict constructed without the field
    legacy_path.write_text(json.dumps({k: v for k, v in data.items() if k != "length_prior"}))
    loaded = UnigramModel.load(str(legacy_path))
    assert loaded.length_prior is None
    assert loaded.token_bias == 0.0
    text = "Taylor Swift wrote songs about love and loss."
    assert loaded.encode(text) == model.encode(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_bias": 1.0},  # penalty without declaring the geometric prior
        {"length_prior": "eow", "token_bias": 1.0},  # eow learns its own parameter
        {"length_prior": "geometric", "token_bias": -1.0},  # a negative penalty would reward long paths
    ],
)
def test_invalid_prior_configs_are_rejected(kwargs):
    """Inconsistent length-prior settings must fail at config time, not silently train a different model."""
    with pytest.raises(ValueError):
        UnigramTrainerConfig(additional_vocab_size=ADDITIONAL_VOCAB, **kwargs)


# ---- 2. EOW ----


def test_eow_probabilities_normalize_with_eow_outcome(train, tmp_path):
    """Pieces plus the EOW outcome form one categorical: they sum to 1 under the plain-ratio M-step. With the
    digamma M-step (E[log p] under a Dirichlet posterior) the sum is below 1. pre_final_vocab_factor=1.0 so that no
    token is pruned after the last M-step, which would remove mass from the sum."""
    exact = train(length_prior="eow", m_step_digamma=False, pre_final_vocab_factor=1.0)
    total = sum(math.exp(t.log_prob) for t in exact.tokens.values()) + math.exp(exact.length_prior["eow_log_prob"])
    assert abs(total - 1.0) < NORMALIZATION_TOL

    digamma = train(length_prior="eow", m_step_digamma=True, pre_final_vocab_factor=1.0)
    eow_log_prob = digamma.length_prior["eow_log_prob"]
    assert math.isfinite(eow_log_prob)
    assert sum(math.exp(t.log_prob) for t in digamma.tokens.values()) + math.exp(eow_log_prob) < 1.0

    path = exact.save(str(tmp_path / "eow.json.gz"))
    reloaded = UnigramModel.load(path)
    assert reloaded.length_prior == exact.length_prior
    assert reloaded.token_bias == 0.0  # EOW is already folded into the piece probabilities


def test_eow_uses_no_more_tokens_than_no_prior(train, swift_text):
    """Every piece probability carries the factor (1 - phi_eow) < 1, i.e. an implicit per-token penalty
    log(1 - phi_eow) < 0, so a path with m tokens is penalized by m*log(1 - phi_eow). Segmentations with fewer
    tokens are therefore preferred and the encoded training text must not get longer than without the prior."""
    none_len = len(train(m_step_digamma=False).encode(swift_text))
    eow_len = len(train(length_prior="eow", m_step_digamma=False).encode(swift_text))
    assert eow_len <= none_len


# ---- 3. geometric ----


def test_geometric_prior_is_saved_and_applied_at_inference(train, swift_text, tmp_path):
    """The lambda used in training is stored on the model, survives save/load, and encode subtracts it on every
    arc: encoding equals that of a prior-free model whose log_probs are shifted by -lambda."""
    model = train(length_prior="geometric", token_bias=GEO_LAMBDA)
    assert model.length_prior == {"type": "geometric", "lambda": GEO_LAMBDA}
    loaded = UnigramModel.load(model.save(str(tmp_path / "geo.json.gz")))
    assert loaded.length_prior == {"type": "geometric", "lambda": GEO_LAMBDA}
    assert loaded.metadata["config"]["token_bias"] == GEO_LAMBDA

    shifted = _shifted_copy(loaded, -GEO_LAMBDA)
    assert shifted.length_prior is None
    assert loaded.encode(swift_text) == shifted.encode(swift_text)
    assert loaded.encode(swift_text) == model.encode(swift_text)


def test_geometric_large_lambda_picks_fewest_tokens():
    """Segmentations of "abc" by probability: ab+c = 0.2*0.3 = 0.06, a+b+c = 0.027, abc = 0.02. Without a penalty
    ab+c wins. With lambda=10 each token costs 10 nats, so the single token abc (-3.9 - 10) beats the two-token ab+c
    (-2.8 - 20); the encoder must pick the segmentation with the fewest tokens."""
    pretokenizer = get_pretokenizer(BYTES_PRETOKENIZER_NAME)
    ids = {c: pretokenizer.encode_text(c)[0].atomic_token_ids[0] for c in "abc"}

    def tok(i, s, p, required=False):
        return UnigramToken(
            id=i, atomic_tokens=token_array([ids[c] for c in s]), log_prob=math.log(p), required=required
        )

    tokens = [
        tok(0, "a", 0.3, True),
        tok(1, "b", 0.3, True),
        tok(2, "c", 0.3, True),
        tok(3, "ab", 0.2),
        tok(4, "abc", 0.02),
    ]
    plain = UnigramModel(pretokenizer, copy.deepcopy(tokens))
    assert plain.encode("abc") == [3, 2]  # 0.06 > 0.027 > 0.02

    penalized = UnigramModel(
        pretokenizer, copy.deepcopy(tokens), length_prior={"type": "geometric", "lambda": HUGE_LAMBDA}
    )
    assert penalized.encode("abc") == [4]  # 1 token: log 0.02 - 10 beats 2 tokens: log 0.06 - 20


# ---- 4. pruning ----


@pytest.fixture
def prune_setup():
    """Atomic '0','1' plus the redundant token '00'. Corpus: '00' x3 and '1' x2, so Viterbi uses '00' 3 times and
    its only alternative is ['0','0'] (k=2). Target size 2 forces '00' to be pruned, which exposes its loss."""
    pretokenizer = get_pretokenizer(BYTES_PRETOKENIZER_NAME)
    zero = pretokenizer.encode_text("0")[0].atomic_token_ids[0]
    one = pretokenizer.encode_text("1")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([zero]), log_prob=math.log(0.2), required=True),
        UnigramToken(id=1, atomic_tokens=token_array([zero, zero]), log_prob=math.log(0.6)),
        UnigramToken(id=2, atomic_tokens=token_array([one]), log_prob=math.log(0.2), required=True),
    ]
    corpus = [(token_array([zero, zero]), 3), (token_array([one]), 2)]
    return pretokenizer, tokens, corpus


def _pruned_loss(setup, monkeypatch, **cfg_kwargs) -> float:
    pretokenizer, tokens, corpus = setup
    trainer = UnigramTrainer(pretokenizer, corpus, UnigramTrainerConfig(additional_vocab_size=0, **cfg_kwargs))
    logged = []
    monkeypatch.setattr(trainer, "log_examples", lambda items, label="score", n=5: logged.append((label, list(items))))
    model = UnigramModel(pretokenizer, copy.deepcopy(tokens))
    new_model, _, num_pruned, _ = trainer.prune_tokens(model, desired_vocab_size=2)
    assert num_pruned == 1 and 1 not in new_model.tokens
    (items,) = [items for label, items in logged if label == "loss"]
    ((token, loss),) = items
    assert token.id == 1
    return loss


def test_prune_loss_includes_length_term(prune_setup, monkeypatch):
    """With effective log-probs (lambda subtracted per arc) a token with a k-token alternative has loss
    (c/total)*((lp_v - lambda) - (lp_alt - k*lambda)) = loss(lambda=0) + (c/total)*lambda*(k-1).
    Here c=3, total=5 token uses, k=2, so lambda=1 raises the loss by 3/5."""
    loss_0 = _pruned_loss(prune_setup, monkeypatch)
    loss_1 = _pruned_loss(prune_setup, monkeypatch, length_prior="geometric", token_bias=1.0)
    assert loss_1 - loss_0 == pytest.approx(3 / 5 * 1.0 * (2 - 1), abs=1e-12)


def test_prune_loss_eow_total_includes_eow_count(prune_setup, monkeypatch):
    """With "eow" every probability in the token-vs-alternative comparison is normalized by total + N_pretokens,
    where N_pretokens = 5 here. Expected loss computed by hand from the counts (token '00': 3, '0': 0, k=2)."""
    c, n_pretokens, total_uses = 3.0, 5.0, 5.0
    total = total_uses + n_pretokens
    lp_token = math.log(c / total)
    lp_alt = 2 * math.log((0 + c) / (total + c * (2 - 1)))
    expected = (c / total) * (lp_token - lp_alt)
    assert _pruned_loss(prune_setup, monkeypatch, length_prior="eow") == pytest.approx(expected, abs=1e-12)
