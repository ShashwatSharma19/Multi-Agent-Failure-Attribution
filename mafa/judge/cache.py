"""A caching decorator over any :class:`Judge`.

Keyed on ``sha256(model_id | system | user | max_tokens | temperature)``, backed
by SQLite. This is the single feature that makes the free cloud tiers workable:

* **Re-runs cost zero quota** — a repeated call hits the cache instead of the API.
* **Runs become deterministic and reproducible** — identical inputs always return
  the same cached text, even across days when a provider silently rotates models.
* **A rate-limit lockout mid-run resumes** — already-computed logs stay cached.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from mafa.judge.base import Judge


def _key(model_id: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    h = hashlib.sha256()
    for part in (model_id, system, user, str(max_tokens), str(temperature)):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")  # field separator so concatenation is unambiguous
    return h.hexdigest()


class CachingJudge(Judge):
    def __init__(self, inner: Judge, db_path: str | Path = ".cache/judge.sqlite"):
        self.inner = inner
        self.model_id = inner.model_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        key = _key(self.model_id, system, user, max_tokens, temperature)
        row = self._conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]

        self.misses += 1
        response = self.inner.complete(system, user, max_tokens, temperature)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, model, response) VALUES (?, ?, ?)",
            (key, self.model_id, response),
        )
        self._conn.commit()
        return response

    def close(self) -> None:
        self._conn.close()
