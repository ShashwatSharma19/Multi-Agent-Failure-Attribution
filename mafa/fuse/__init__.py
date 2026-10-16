"""Phase 3 fusion: combine the direct LLM vote with the spectrum suspiciousness."""

from mafa.fuse.fusion import blend, fuse, normalize
from mafa.fuse.split import split_traces, tune_weight

__all__ = ["fuse", "normalize", "blend", "split_traces", "tune_weight"]
