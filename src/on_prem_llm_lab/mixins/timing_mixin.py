"""TTFT + TPOT measurement around a streaming-token iterable (T-2.2).

Bridges PLAN §6.2's ``ttft_ms`` + ``tpot_ms`` :class:`BackendRunResult`
fields to the three concrete back-ends. Each back-end produces a
token-by-token iterable (HF ``TextIteratorStreamer``, AirLLM streaming
generate, Anthropic ``messages.stream`` events) and hands it to
:meth:`TimingMixin.time_token_stream`; the mixin wraps the iteration
with ``clock()`` calls and returns a :class:`TimingMeasurement`.

The mixin is intentionally generator-agnostic — it does not care what
the token is, only that one ``next()`` corresponds to one output token.
A ``clock`` parameter is injectable so tests drive the iteration with a
deterministic timestamp sequence.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingMeasurement:
    """Result of timing a token stream.

    ``ttft_ms`` is ``None`` when no tokens were produced.
    ``tpot_ms`` is ``None`` when fewer than two tokens were produced —
    TPOT is undefined with a single token (no decode interval to divide).
    """

    ttft_ms: float | None
    tpot_ms: float | None
    n_tokens: int


class TimingMixin:
    """Stateless mixin: measures TTFT + TPOT around a token iterable.

    Concrete back-ends inherit this alongside their own methods. Usage::

        stream = backend.start_generation(prompt)   # Iterable[Token]
        measurement = self.time_token_stream(stream)
        # measurement.ttft_ms / .tpot_ms / .n_tokens

    The mixin consumes the iterable via ``next()`` only — if the caller
    also needs the tokens themselves, ``itertools.tee`` or a capturing
    list comprehension is the right wrapper.
    """

    def time_token_stream(
        self,
        stream: Iterable[object],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> TimingMeasurement:
        """Walk ``stream``, measure TTFT to first token + TPOT over the rest.

        Wall-time math: ``t0 = clock()`` before pulling any token;
        ``t_first = clock()`` right after the first ``next()`` returns;
        ``t_last = clock()`` right after the last ``next()`` returns. Then:

        * ``ttft_ms = (t_first - t0) * 1000``
        * ``tpot_ms = (t_last - t_first) * 1000 / (n - 1)`` for ``n >= 2``.
        """
        it: Iterator[object] = iter(stream)
        t0 = clock()
        try:
            next(it)
        except StopIteration:
            return TimingMeasurement(ttft_ms=None, tpot_ms=None, n_tokens=0)
        t_first = clock()
        n = 1
        t_last = t_first
        for _ in it:
            n += 1
            t_last = clock()
        ttft_ms = (t_first - t0) * 1000.0
        if n < 2:
            return TimingMeasurement(ttft_ms=ttft_ms, tpot_ms=None, n_tokens=1)
        tpot_ms = ((t_last - t_first) * 1000.0) / (n - 1)
        return TimingMeasurement(ttft_ms=ttft_ms, tpot_ms=tpot_ms, n_tokens=n)


__all__ = ["TimingMeasurement", "TimingMixin"]
