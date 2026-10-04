from mafa.judge.cache import CachingJudge
from mafa.judge.ratelimit import RateLimitedJudge
from tests.conftest import MockJudge


def test_caching_judge_calls_backend_once(tmp_path):
    inner = MockJudge(["hello"])
    cached = CachingJudge(inner, db_path=tmp_path / "c.sqlite")
    a = cached.complete("sys", "user")
    b = cached.complete("sys", "user")
    assert a == b == "hello"
    assert len(inner.calls) == 1  # second call served from cache
    assert cached.hits == 1 and cached.misses == 1


def test_cache_key_separates_different_prompts(tmp_path):
    inner = MockJudge(["r1", "r2"])
    cached = CachingJudge(inner, db_path=tmp_path / "c.sqlite")
    assert cached.complete("sys", "A") == "r1"
    assert cached.complete("sys", "B") == "r2"
    assert len(inner.calls) == 2


def test_cache_persists_across_instances(tmp_path):
    db = tmp_path / "c.sqlite"
    first = CachingJudge(MockJudge(["once"]), db_path=db)
    first.complete("sys", "user")
    first.close()
    inner2 = MockJudge(["SHOULD-NOT-BE-USED"])
    second = CachingJudge(inner2, db_path=db)
    assert second.complete("sys", "user") == "once"
    assert len(inner2.calls) == 0  # answered from the persisted cache


def test_ratelimit_retries_then_succeeds():
    inner = MockJudge(["ok"])
    inner.fail_times = 2  # first two calls raise JudgeError
    sleeps = []
    judge = RateLimitedJudge(inner, rpm=0, base_delay=1.0, sleep=sleeps.append)
    assert judge.complete("s", "u") == "ok"
    assert sleeps == [1.0, 2.0]  # exponential backoff between the 3 attempts


def test_ratelimit_gives_up_after_max_retries():
    import pytest

    from mafa.judge.base import JudgeError

    inner = MockJudge(["never"])
    inner.fail_times = 99
    judge = RateLimitedJudge(inner, rpm=0, max_retries=2, sleep=lambda _: None)
    with pytest.raises(JudgeError):
        judge.complete("s", "u")
