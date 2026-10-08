from mafa.judge.accounting import AccountingJudge, estimate_tokens, unwrap
from mafa.judge.cache import CachingJudge
from tests.conftest import MockJudge


def test_accounting_counts_calls_and_tokens():
    inner = MockJudge(["hello there"])
    acc = AccountingJudge(inner)
    acc.complete("system text", "user text")
    assert acc.calls == 1
    assert acc.prompt_tokens == estimate_tokens("system text") + estimate_tokens("user text")
    assert acc.completion_tokens == estimate_tokens("hello there")
    assert acc.tokens == acc.prompt_tokens + acc.completion_tokens


def test_accounting_reset():
    acc = AccountingJudge(MockJudge(["a", "b"]))
    acc.complete("s", "u")
    acc.reset()
    assert acc.calls == 0 and acc.tokens == 0


def test_unwrap_finds_inner_wrapper(tmp_path):
    inner = MockJudge(["x"])
    stack = AccountingJudge(CachingJudge(inner, db_path=tmp_path / "c.sqlite"))
    assert isinstance(unwrap(stack, CachingJudge), CachingJudge)
    assert isinstance(unwrap(stack, AccountingJudge), AccountingJudge)
    assert unwrap(inner, CachingJudge) is None
