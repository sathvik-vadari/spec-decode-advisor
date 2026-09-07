"""Turning one speculative run and two timings into a recommendation.

The procedure experiment 04 validated:

1.  Time the target's forward pass at 1..9 tokens: `c_verify`.
2.  Time each draft's one-token pass against the target's: `c_draft`.
3.  Run the workload once without speculation and once at k=1 per draft. The
    k=1 run gives acceptance `p` (per position, so depth does not matter --
    finding 2) and one measured speedup, which pins the fixed cost.
4.  Search depths and drafts under the resulting cost models.

This module is the analysis half: it takes run results and timings and returns
a report. It touches no model, so it is tested on synthetic inputs. `cli.py` is
the half that runs things.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .acceptance import bootstrap_ci, filter_rounds, pooled_acceptance
from .calibration import PassLatency, draft_pass_ratio, verification_cost
from .harness import RunResult, to_prompt_rounds
from .model import Candidate, CostModel, Recommendation, fit_fixed_cost, recommend

PIN_K = 1


@dataclass(frozen=True, slots=True)
class DomainAcceptance:
    p: float
    ci: tuple[float, float]
    n_rounds: int
    n_prompts: int


@dataclass(frozen=True, slots=True)
class DraftReport:
    name: str
    c_draft: float
    c_verify: float
    fixed_cost: float
    p: float
    p_ci: tuple[float, float]
    by_domain: dict[str, DomainAcceptance]
    measured_speedup_k1: float
    baseline_tokens_per_s: float
    lossless_mismatches: int
    n_prompts: int

    @property
    def cost(self) -> CostModel:
        return CostModel(draft_cost_ratio=self.c_draft + self.c_verify, fixed_cost=self.fixed_cost)

    def candidate(self, domain: str | None = None) -> Candidate:
        p = self.p if domain is None else self.by_domain[domain].p
        return Candidate(self.name, p, self.cost)


@dataclass(frozen=True, slots=True)
class Report:
    target: str
    target_pass_ms: float
    c_verify: float
    drafts: tuple[DraftReport, ...]
    overall: Recommendation
    by_domain: dict[str, Recommendation]
    max_k: int
    target_pass_ms_by_draft: dict[str, float]

    @property
    def target_drift(self) -> float:
        """max / min of the target's one-token pass across draft sessions, minus 1.

        The target is timed again for every draft, so this is free, and it is
        the check that the machine stayed the same while the tool ran. The
        first 7B replication of this tool measured the target at 89 ms where
        experiment 03 had 57 ms three days earlier, and every ratio moved with
        it; ratios are only comparable across runs if this stays small."""
        xs = list(self.target_pass_ms_by_draft.values())
        return max(xs) / min(xs) - 1.0 if len(xs) > 1 and min(xs) > 0 else 0.0


def analyse_draft(
    name: str,
    runs: Sequence[RunResult],
    draft_latency: PassLatency,
    target_latency: PassLatency,
) -> DraftReport:
    """One draft's report from its k=0 and k=1 runs over the same prompts."""
    base = {r.prompt_id: r for r in runs if r.k == 0}
    spec = [r for r in runs if r.k == PIN_K]
    if not base or not spec:
        raise ValueError("need both k=0 and k=1 runs")
    missing = [r.prompt_id for r in spec if r.prompt_id not in base]
    if missing:
        raise ValueError(f"k=1 runs without a baseline: {missing[:3]}")

    c_verify = verification_cost(target_latency.seconds)
    c_draft = draft_pass_ratio(draft_latency, target_latency)
    slope = c_draft + c_verify

    rounds = to_prompt_rounds(spec)
    p = pooled_acceptance(rounds)
    p_ci = bootstrap_ci(rounds, pooled_acceptance, seed=1)
    speedup = sum(r.tokens_per_s / base[r.prompt_id].tokens_per_s for r in spec) / len(spec)
    fixed = fit_fixed_cost(PIN_K, p, speedup, slope).fixed_cost

    by_domain = {}
    for dom in sorted({r.domain for r in spec}):
        rd = filter_rounds(rounds, domain=dom)
        by_domain[dom] = DomainAcceptance(
            p=pooled_acceptance(rd),
            ci=bootstrap_ci(rd, pooled_acceptance, seed=2),
            n_rounds=sum(r.n_rounds for r in rd),
            n_prompts=len(rd),
        )

    mismatches = sum(1 for r in spec if r.text != base[r.prompt_id].text)
    base_tps = sum(b.tokens_per_s for b in base.values()) / len(base)
    return DraftReport(
        name=name, c_draft=c_draft, c_verify=c_verify, fixed_cost=fixed,
        p=p, p_ci=p_ci, by_domain=by_domain, measured_speedup_k1=speedup,
        baseline_tokens_per_s=base_tps, lossless_mismatches=mismatches, n_prompts=len(spec),
    )


def build_report(
    target: str,
    target_latency: PassLatency | Sequence[PassLatency],
    drafts: Sequence[DraftReport],
    max_k: int = 8,
) -> Report:
    """`target_latency` is one timing, or one per draft in the same order."""
    lats = [target_latency] if isinstance(target_latency, PassLatency) else list(target_latency)
    if len(lats) not in (1, len(drafts)):
        raise ValueError("give one target timing, or one per draft")
    by_draft = {d.name: lat.one_token * 1e3 for d, lat in zip(drafts, lats if len(lats) > 1 else lats * len(drafts))}
    first = lats[0]
    overall = recommend([d.candidate() for d in drafts], max_k=max_k)
    domains = sorted({dom for d in drafts for dom in d.by_domain})
    by_domain = {
        dom: recommend([d.candidate(dom) for d in drafts if dom in d.by_domain], max_k=max_k)
        for dom in domains
    }
    return Report(
        target=target, target_pass_ms=first.one_token * 1e3,
        c_verify=verification_cost(first.seconds), drafts=tuple(drafts),
        overall=overall, by_domain=by_domain, max_k=max_k, target_pass_ms_by_draft=by_draft,
    )


def _rec(r: Recommendation) -> str:
    if r.draft is None:
        return "do not speculate"
    return f"{r.draft} at k={r.k}, predicted {r.speedup:.2f}x"


def render(rep: Report) -> str:
    out = []
    out.append(f"target {rep.target}")
    out.append(f"  one-token pass {rep.target_pass_ms:.1f} ms; each extra verified token costs "
               f"{rep.c_verify:.2f} of a pass" + ("  (verification is not free here)" if rep.c_verify > 0.1 else ""))
    if rep.target_drift > 0.10:
        passes = ", ".join(f"{n} {ms:.0f} ms" for n, ms in rep.target_pass_ms_by_draft.items())
        out.append(f"  WARNING: the target's pass time moved {rep.target_drift:.0%} between draft sessions "
                   f"({passes}). The machine did not hold still; compare drafts with care.")
    out.append("")
    out.append(f"{'draft':>8} {'c_draft':>8} {'slope':>6} {'fixed':>6} {'p':>6} {'95% CI':>16} "
               f"{'k=1 meas':>9} {'best k':>7} {'pred':>6} {'lossless':>9}")
    for d in rep.drafts:
        k, s = d.cost.best_depth(d.p, rep.max_k)
        out.append(f"{d.name:>8} {d.c_draft:>8.3f} {d.c_draft + d.c_verify:>6.3f} {d.fixed_cost:>6.3f} "
                   f"{d.p:>6.3f} [{d.p_ci[0]:.3f}, {d.p_ci[1]:.3f}] {d.measured_speedup_k1:>9.2f} "
                   f"{k:>7} {s:>6.2f} {d.n_prompts - d.lossless_mismatches:>4}/{d.n_prompts:<4}")
    low = [d for d in rep.drafts if d.fixed_cost < 0.95]
    if low:
        names = ", ".join(f"{d.name} {d.fixed_cost:.2f}" for d in low)
        out.append(f"  WARNING: pinned fixed cost below one target pass ({names}). The k=1 run paid less than "
                   "the sum of its timed parts, so the pass timings overstate in-loop cost -- usually a busy "
                   "host or a throttled GPU. Predictions at other depths will be pessimistic.")
    out.append("")
    if len(rep.by_domain) > 1:
        out.append("by domain")
        for dom, r in rep.by_domain.items():
            ps = "  ".join(f"{d.name} p={d.by_domain[dom].p:.3f}" for d in rep.drafts if dom in d.by_domain)
            out.append(f"  {dom:>10}  {ps}  -> {_rec(r)}")
        out.append("")
    out.append(f"recommendation: {_rec(rep.overall)}")
    if rep.overall.draft is not None and len(rep.overall.ranking) > 1:
        runner = rep.overall.ranking[1]
        out.append(f"  runner-up: {runner[0]} at k={runner[1]}, {runner[2]:.2f}x")
    mism = sum(d.lossless_mismatches for d in rep.drafts)
    if mism:
        out.append(f"  note: {mism} speculative outputs differed from the baseline under greedy decoding. "
                   "Expected -- verification runs a wider batch and near-tied argmaxes flip. Golden-output tests will see this.")
    return "\n".join(out)


def to_json(rep: Report) -> dict:
    def rec(r: Recommendation) -> dict:
        return {"draft": r.draft, "k": r.k, "speedup": r.speedup, "ranking": [list(t) for t in r.ranking]}
    return {
        "target": rep.target,
        "target_pass_ms": rep.target_pass_ms,
        "target_pass_ms_by_draft": rep.target_pass_ms_by_draft,
        "target_drift": rep.target_drift,
        "c_verify": rep.c_verify,
        "max_k": rep.max_k,
        "drafts": [
            {**{k: v for k, v in asdict(d).items() if k != "by_domain"},
             "slope": d.c_draft + d.c_verify,
             "by_domain": {dom: asdict(a) for dom, a in d.by_domain.items()}}
            for d in rep.drafts
        ],
        "overall": rec(rep.overall),
        "by_domain": {dom: rec(r) for dom, r in rep.by_domain.items()},
    }
