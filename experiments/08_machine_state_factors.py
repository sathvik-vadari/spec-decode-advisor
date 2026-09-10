"""What moves the pass-cost curve: thermal history, memory pressure, or neither?

Finding 14: the same pair went from 1.15x to 1.76x between Sep 4 and Sep 9. The
Sep 4 curve had no free region -- the second and third verified tokens each cost
~12 ms -- and the Sep 9 curve was flat to four tokens. Low Power Mode is a proven
cause of a slow curve (finding 12); the power source alone is not (experiment 07
on battery at 93% kept the free region). Sep 4's run followed a 49-minute GPU
sweep, with other applications holding memory. So two candidates remain that
this machine can manipulate: thermal history and memory pressure.

Design: a 2x2 with conditions run in this order, GPU cold at the start (idle
for 30 minutes):

    cold / normal        the reference
    cold / pressured     host memory pushed to ~12% free by allocating and
                         re-touching pages, model already resident
    burn                 15 minutes of sustained 4096x4096 matmuls, its own
                         throughput logged per 30 s -- a throttling GPU shows
                         up here directly, before any pass is timed
    hot / normal         immediately after the burn
    hot / pressured      same, with the allocation back

Per condition: the 7B pass curve at T = 1..9, the 0.5B draft's pass, and on ten
prompts (two per domain) plain decoding and k=2, for a speedup and acceptance.

Predictions, written before the run:

P1. Memory pressure alone does not remove the free region: V(2) and V(3) stay
    within 10% of V(1) in cold/pressured. The weights are resident GPU buffers;
    host pressure should not reach a pass that touches nothing else.
P2. The burn's matmul throughput falls by at least 10% between its first and
    last minute -- the M4 throttles under sustained load -- and the hot curve
    loses its free region: V(2) >= 1.10 in hot/normal. If throughput does not
    fall, thermal history is ruled out as Sep 4's cause on this machine.
P3. In whichever condition reproduces Sep 4's curve (V(2) >= 1.15), the k=2
    speedup on the ten prompts falls from its cold/normal value by at least
    0.25. If no condition reproduces the curve, the cause is not among these
    two and the result is a null.
P4. Acceptance on the ten prompts is identical across all four conditions.
    Determinism check, not a finding.

Report is checkpointed after every condition. Conditions are recorded.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spec_decode_advisor.acceptance import pooled_acceptance  # noqa: E402
from spec_decode_advisor.calibration import time_passes  # noqa: E402
from spec_decode_advisor.harness import Harness, to_prompt_rounds  # noqa: E402
from spec_decode_advisor.prompts import PROMPTS  # noqa: E402

TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TOKENS = tuple(range(1, 10))
SUBSET = [PROMPTS[i] for i in (0, 1, 6, 7, 12, 13, 18, 19, 24, 25)]   # two per domain
K = 2
BURN_S = 900
PRESSURE_TARGET_FREE_PCT = 15   # 12% killed the first speculative generation with a GPU timeout
PRESSURE_CAP_GB = 10
OUT = ROOT / "results" / "08_machine_state_factors.json"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def conditions() -> dict:
    return {"lowpowermode": sh("pmset -g | awk '/lowpowermode/ {print $2}'"),
            "power": sh("pmset -g batt | head -1"),
            "battery": sh("pmset -g batt | sed -n 2p | awk '{print $3, $4}'"),
            "memory_free": sh("memory_pressure | grep -i 'system-wide'"),
            "therm": sh("pmset -g therm | tr '\\n' ' '")}


def free_pct() -> float:
    s = sh("memory_pressure | grep -i 'system-wide'")
    try:
        return float(s.split(":")[1].strip().rstrip("%"))
    except Exception:  # noqa: BLE001
        return float("nan")


class Pressure:
    """Hold host memory until the system reports ~target% free, and keep it hot."""

    def __init__(self, target_pct: float, cap_gb: float):
        self.target, self.cap = target_pct, cap_gb
        self.chunks: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> dict:
        # Random bytes, not ones: the first attempt allocated 10 GB of np.ones in
        # 2.5 s and free memory went *up* -- identical pages compress to nothing
        # under the macOS memory compressor. Incompressible data has to be paid for.
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        while free_pct() > self.target and len(self.chunks) * 0.5 < self.cap:
            c = rng.random(512 * 1024 * 1024 // 8, dtype=np.float64)   # 512 MB, incompressible
            self.chunks.append(c)
        self._thread = threading.Thread(target=self._touch, daemon=True)
        self._thread.start()
        return {"allocated_gb": len(self.chunks) * 0.5, "free_pct_after": free_pct(), "seconds": time.perf_counter() - t0}

    def _touch(self) -> None:
        while not self._stop.is_set():
            for c in self.chunks:
                c[::4096] += 1.0          # one write per page keeps it resident
            self._stop.wait(2.0)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.chunks.clear()
        gc.collect()


def burn(seconds: int) -> list[dict]:
    """Sustained GPU load; returns matmuls per second per 30-second window."""
    a = mx.random.normal((4096, 4096)).astype(mx.float16)
    mx.eval(a)
    log, t0, w0, n = [], time.perf_counter(), time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        b = a @ a
        mx.eval(b)
        n += 1
        if time.perf_counter() - w0 >= 30:
            log.append({"t_s": time.perf_counter() - t0, "matmul_per_s": n / (time.perf_counter() - w0)})
            print(f"    burn {log[-1]['t_s']:5.0f}s: {log[-1]['matmul_per_s']:.1f} matmul/s", flush=True)
            w0, n = time.perf_counter(), 0
    return log


def _retry(fn, label: str, attempts: int = 2):
    """GPU command-buffer timeouts have hit this project three times, always with
    memory tight. One kills a generation and leaves the process fine. Retry once,
    then give up on that item and count it -- the count is data."""
    for i in range(attempts):
        try:
            return fn(), i
        except RuntimeError as e:
            print(f"    {label}: {str(e).splitlines()[0][:80]}" + (" -- retrying" if i + 1 < attempts else " -- skipped"), flush=True)
    return None, attempts


def measure(h: Harness, label: str) -> dict:
    print(f"[{label}] conditions {json.dumps(conditions())}", flush=True)
    ids = h.encode_ids(PROMPTS[6])
    timeouts = 0
    t, n1 = _retry(lambda: time_passes(h.target, ids, tokens=TOKENS, repeats=15), "target timing")
    d, n2 = _retry(lambda: time_passes(h.draft, ids, tokens=(1,), repeats=15), "draft timing")
    if t is None or d is None:
        raise RuntimeError(f"{label}: pass timing failed twice")
    timeouts += n1 + n2
    curve = {n: s / t.one_token for n, s in t.seconds.items()}
    print(f"  pass {t.one_token * 1e3:.1f} ms; V: " + " ".join(f"{n}:{v:.2f}" for n, v in curve.items() if n > 1), flush=True)

    _, n3 = _retry(lambda: h.run(SUBSET[0], K), "warm-up"); timeouts += n3
    results, skipped = [], []
    for p in SUBSET:
        pair, n4 = _retry(lambda: (h.run(p, 0), h.run(p, K)), p.prompt_id)
        timeouts += n4
        if pair is None:
            skipped.append(p.prompt_id)
        else:
            results.extend(pair)
    base = {r.prompt_id: r.tokens_per_s for r in results if r.k == 0}
    spec = [r for r in results if r.k == K]
    speedup = sum(r.tokens_per_s / base[r.prompt_id] for r in spec) / len(spec)
    p_acc = pooled_acceptance(to_prompt_rounds(spec))
    print(f"  baseline {sum(base.values()) / len(base):.1f} tok/s; k={K} speedup {speedup:.3f}; p {p_acc:.3f}", flush=True)
    return {"conditions": conditions(), "gpu_timeouts": timeouts, "skipped_prompts": skipped,
            "target_pass_ms": t.one_token * 1e3,
            "target_latency_ms": {str(n): s * 1e3 for n, s in t.seconds.items()},
            "pass_curve": {str(n): v for n, v in curve.items()},
            "draft_pass_ms": d.one_token * 1e3, "c_draft": d.one_token / t.one_token,
            "baseline_tokens_per_s": sum(base.values()) / len(base), "speedup_k2": speedup, "acceptance_k2": p_acc}


def main() -> None:
    t0 = time.perf_counter()
    payload = {"config": {"target": TARGET, "draft": DRAFT, "tokens": list(TOKENS), "k": K, "n_prompts": len(SUBSET),
                          "burn_s": BURN_S, "pressure_target_free_pct": PRESSURE_TARGET_FREE_PCT},
               "start_conditions": conditions(), "cells": {}, "burn_log": [], "pressure": {}}

    def checkpoint():
        payload["elapsed_s"] = time.perf_counter() - t0
        OUT.write_text(json.dumps(payload, indent=2))

    h = Harness(TARGET, DRAFT, max_tokens=128).load()

    payload["cells"]["cold_normal"] = measure(h, "cold/normal"); checkpoint()

    pr = Pressure(PRESSURE_TARGET_FREE_PCT, PRESSURE_CAP_GB)
    payload["pressure"]["cold"] = pr.start(); print(f"  pressure: {payload['pressure']['cold']}", flush=True)
    payload["cells"]["cold_pressured"] = measure(h, "cold/pressured"); checkpoint()
    pr.stop()

    print(f"burning for {BURN_S}s", flush=True)
    payload["burn_log"] = burn(BURN_S); checkpoint()

    payload["cells"]["hot_normal"] = measure(h, "hot/normal"); checkpoint()

    pr = Pressure(PRESSURE_TARGET_FREE_PCT, PRESSURE_CAP_GB)
    payload["pressure"]["hot"] = pr.start(); print(f"  pressure: {payload['pressure']['hot']}", flush=True)
    payload["cells"]["hot_pressured"] = measure(h, "hot/pressured"); checkpoint()
    pr.stop()

    cells = payload["cells"]
    ref = cells["cold_normal"]
    bl = payload["burn_log"]
    payload["checks"] = {
        "P1_pressure_keeps_free_region": all(cells["cold_pressured"]["pass_curve"][str(n)] <= 1.10 for n in (2, 3)),
        "P2a_burn_throughput_falls_10pct": bool(bl) and bl[-1]["matmul_per_s"] <= 0.9 * bl[0]["matmul_per_s"],
        "P2b_hot_loses_free_region": cells["hot_normal"]["pass_curve"]["2"] >= 1.10,
        "P3_speedup_falls_where_curve_reproduces": any(
            c["pass_curve"]["2"] >= 1.15 and ref["speedup_k2"] - c["speedup_k2"] >= 0.25 for c in cells.values()),
        "any_cell_reproduces_sep4_curve": any(c["pass_curve"]["2"] >= 1.15 for c in cells.values()),
        "P4_acceptance_identical": len({round(c["acceptance_k2"], 3) for c in cells.values()}) == 1,
    }
    checkpoint()

    print()
    print(f"{'cell':>15} {'pass ms':>8} {'V2':>5} {'V3':>5} {'V5':>5} {'V9':>5} {'c_draft':>8} {'base tok/s':>10} {'k=2 x':>6} {'p':>6}")
    for name, c in cells.items():
        v = c["pass_curve"]
        print(f"{name:>15} {c['target_pass_ms']:>8.1f} {v['2']:>5.2f} {v['3']:>5.2f} {v['5']:>5.2f} {v['9']:>5.2f} "
              f"{c['c_draft']:>8.3f} {c['baseline_tokens_per_s']:>10.1f} {c['speedup_k2']:>6.3f} {c['acceptance_k2']:>6.3f}")
    if bl:
        print(f"burn: {bl[0]['matmul_per_s']:.1f} -> {bl[-1]['matmul_per_s']:.1f} matmul/s over {bl[-1]['t_s']:.0f}s")
    for k, ok in payload["checks"].items():
        print(f"{k}: {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
