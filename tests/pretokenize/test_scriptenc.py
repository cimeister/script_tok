import pytest

from script_bpe.pretokenize import export_pretokenizer, get_pretokenizer
from script_bpe.utils import TokenSeq, token_array


def test_se_tokenize(script_encoding_pretokenizer):
    text = "abc def"
    tokenized = script_encoding_pretokenizer.pretokenize(text)
    assert isinstance(tokenized, list)
    assert all(isinstance(group, TokenSeq) for group in tokenized)

    token_ids = sum(tokenized, token_array([]))
    detokenized = script_encoding_pretokenizer.decode(token_ids)
    assert isinstance(detokenized, str)
    assert detokenized == text


# string, v1 groups, v2 groups
TEST_CASES = [
    # 1. Latin "Hello ", Arabic "العالمية", spaces, Emoji "❤️"
    ("Hello \u0627\u0644\u0639\u0627\u0644\u0645\u064a  \u2764", 4, 4),
    # 2. Latin "Hello ", Arabic "العالمية", ZWJ+Emoji "‍❤️"
    ("Hello \u0627\u0644\u0639\u0627\u0644\u0645\u064a\u200c❤️", 3, 3),
    # 3. Latin "Hello ", Arabic "العالمية", space, Emoji "🌍"
    ("Hello \u0627\u0644\u0639\u0627\u0644\u0645\u064a \u0020🌍", 4, 4),
    # 4. Latin "abc ", Arabic "مَبْنِي" (with diacritics), spaces "  ", Latin "xyz"
    ("abc \u0645\u0650\u0628\u0646\u064a  xyz", 4, 4),
    # 5. Arabic "رينر" (with diacritic), space+Emoji "❤️"
    ("\u0631\u064a\u0646\u0650\u0631 ❤️", 2, 3),
    # 6. Pure Arabic text only
    ("\u0627\u0639\u062f\u0644\u0639\u0645\u062f\u0631\u0648\u0632", 1, 1),
    # 7. Tamil "வணக்கம்", space, Emoji "❤️"
    ("வணக்கம் ❤️", 2, 3),
    # 8. Mixed chinese
    ("欢迎歡迎来到來到中国中國一起学习學習古字𠀋𠀍𠀎", 1, 1),
    # 9. Numbers do not combine with spaces
    ("1234 5678", 3, 3),
    # 10. Punctuation does combine
    ("x = a + b", 5, 5),
    # 11. Japanese: Han -> Hira -> Kata (Should all merge into one chunk)
    ("漢字ひらがな", 1, 1),
    # 12. Japanese: Hira -> Han -> Kata (Should all merge into one chunk)
    ("ひらがな漢字", 1, 1),
    # 18. Japanese with Space
    (" 日本語 ", 3, 3),  # Space, JPN(Han+Hira), Space
    # 19. Japanese with Number (Should split)
    ("第1番", 3, 3),  # JPN(Han), Number, JPN(Han)
]
PRETOKENIZER_NAMES = ["scriptenc_cb", "scriptenc2_cb"]
VERSION_WITH_TEST_CASES = [(text, expected[i], PRETOKENIZER_NAMES[i]) for text, *expected in TEST_CASES for i in [0, 1]]


@pytest.mark.parametrize("text, n_expected, pretokenizer_name", VERSION_WITH_TEST_CASES)
def test_se_pretokenize_groups(pretokenizer_name, text, n_expected):
    script_encoding_pretokenizer = get_pretokenizer(pretokenizer_name)
    pretokenized_groups = script_encoding_pretokenizer.pretokenize(text)
    redecoded_groups = [script_encoding_pretokenizer.decode(group) for group in pretokenized_groups]
    assert len(pretokenized_groups) == n_expected, (
        f"Expected {n_expected} groups but found {len(pretokenized_groups)} for {text!r}, got {len(pretokenized_groups)} groups: {redecoded_groups}"
    )

    # detokenize and check if we get the original string back
    detokenized = script_encoding_pretokenizer.decode(sum(pretokenized_groups, token_array([])))
    assert detokenized == text, f"Detokenized string {detokenized!r} does not match original {text!r}"


@pytest.mark.parametrize("text", [text for text, *_ in TEST_CASES])
def test_nosplit_pretokenizer(text, script_encoding_nosplit_pretokenizer):
    pretokenized_groups = script_encoding_nosplit_pretokenizer.pretokenize(text)
    assert len(pretokenized_groups) == 1
    assert len(pretokenized_groups[0]) == len(text) * 2


def test_se_hash_identical(script_encoding_pretokenizer):
    pretokenizer1 = get_pretokenizer("scriptenc")
    assert pretokenizer1.hash() == script_encoding_pretokenizer.hash()
    print(pretokenizer1.hash())


def test_se_can_json(script_encoding_pretokenizer):
    export_pretokenizer(script_encoding_pretokenizer)


def test_split_line_breaks_variant():
    # scriptenc_cb_nl: a newline-to-space transition always breaks a group, so
    # indentation stays compositional (the mingram MBPP-collapse fix); the base
    # scriptenc_cb variant is unchanged and still fuses "\n" with indentation.
    fixed = get_pretokenizer("scriptenc_cb_nl")
    base = get_pretokenizer("scriptenc_cb")
    text = "\n    return x"

    fixed_groups = [fixed.decode(g) for g in fixed.pretokenize(text)]
    base_groups = [base.decode(g) for g in base.pretokenize(text)]
    assert fixed_groups[0] == "\n" and fixed_groups[1] == "    ", fixed_groups
    assert base_groups[0] == "\n    ", base_groups

    # Pure runs still group under the fixed variant: blank lines and indentation
    # depth tokens are preserved; only the transition splits.
    assert [fixed.decode(g) for g in fixed.pretokenize("\n\n")][0] == "\n\n"
    assert [fixed.decode(g) for g in fixed.pretokenize("a  b")][1] == "  "

    # Lossless round-trip for both variants on a CRLF/indented mixed sample.
    sample = "def f(x):\r\n    return x\n\n你好 नमस्ते"
    for p in (fixed, base):
        ids = sum(p.pretokenize(sample), token_array([]))
        assert p.decode(ids) == sample
