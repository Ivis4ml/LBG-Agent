"""Discovery campaign · keep iterating on failure (PROPOSAL §13 spirit).

A campaign is K Discovery iterations against the same data splits, where:

  * Each iteration starts from the same baseline strategy.yaml + indicators/
    snapshot (taken at CampaignRunner.__init__ time).
  * **Event memory is wiped** between iterations -- trials.jsonl,
    reflections.jsonl, agent_compute.jsonl, invariant_failures.jsonl, plus
    runs/ and artifacts/sealed/. Each iteration tells its own audit story.
  * **Semantic memory persists** -- accepted_rules.md, failed_directions.md,
    open_questions.md, do_not_repeat.md. Iteration N+1's Editor inherits
    everything iteration N learned.
  * **Alpha cards accumulate** -- the alpha_cards/ directory grows; a
    surviving factor from iteration 0 stays available in iteration 1, and
    the campaign's H1 count is the union across iterations.

The Atari analogy from Jiayi Weng's HL blog: a failed run does not erase
what was learned, the agent reincarnates with its accumulated rulebook and
the next attempt is materially better-informed than the first.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lbg.gate import GateConfig
from lbg.orchestrator.discovery import Discovery, DiscoveryResult
from lbg.orchestrator.role_runner import RoleRunner
from lbg.sealed_vault import SealedVault

logger = logging.getLogger(__name__)


# Files we wipe between iterations. semantic memory (.md) and skills/ +
# alpha_cards/ are intentionally NOT in this list -- their persistence is
# the whole point of running a campaign.
_EVENT_MEMORY_FILENAMES = (
    "trials.jsonl",
    "reflections.jsonl",
    "agent_compute.jsonl",
    "invariant_failures.jsonl",
)


@dataclass
class IterationResult:
    iteration_id: int
    discovery_result: DiscoveryResult
    # Snapshot of how many alpha cards exist at the end of this iteration.
    alpha_cards_total_after: int
    # Just the cards this iteration emitted; the campaign sums these for H1.
    alpha_cards_emitted_this_iteration: int
    sealed_summary: dict[str, Any] | None


@dataclass
class CampaignResult:
    n_iterations: int
    budget_per_iteration: int
    iteration_results: list[IterationResult] = field(default_factory=list)

    @property
    def total_accepted(self) -> int:
        return sum(r.discovery_result.n_accepted for r in self.iteration_results)

    @property
    def total_alpha_cards(self) -> int:
        """Cards emitted across the whole campaign. The H1 counting layer
        (PROPOSAL §4) will further filter by sealed CI; this is the upper
        bound on cards eligible to count."""
        return sum(r.alpha_cards_emitted_this_iteration for r in self.iteration_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "budget_per_iteration": self.budget_per_iteration,
            "total_accepted": self.total_accepted,
            "total_alpha_cards": self.total_alpha_cards,
            "iterations": [
                {
                    "iteration_id": r.iteration_id,
                    "n_trials_attempted": r.discovery_result.n_trials_attempted,
                    "n_accepted": r.discovery_result.n_accepted,
                    "n_rejected": r.discovery_result.n_rejected,
                    "n_aborted": r.discovery_result.n_aborted,
                    "n_invariant_failures": r.discovery_result.n_invariant_failures,
                    "alpha_cards_emitted": r.alpha_cards_emitted_this_iteration,
                    "alpha_cards_total_after": r.alpha_cards_total_after,
                    "sealed_summary": r.sealed_summary,
                }
                for r in self.iteration_results
            ],
        }


class CampaignRunner:
    """Run K Discovery iterations with persisted semantic memory and cards.

    Designed to keep iterating on failure: the user's framing
    "如果失败就一直迭代，类似 jiayi weng 打游戏" maps to (a) wipe the per-
    iteration event memory so each Discovery tells its own story, (b) carry
    over what each iteration learned in its .md files so the next one
    starts smarter than the last.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        runner: RoleRunner | None = None,
        gate_config: GateConfig | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runner = runner
        # Inject the gate per-iteration. None falls back to Discovery's strict
        # default (PROPOSAL §7). For experiment-stage campaigns that want a
        # non-zero accept rate, pass GateConfig.permissive() explicitly.
        self.gate_config = gate_config

        # Snapshot the baseline at __init__ time. We re-apply this snapshot
        # before each iteration so failed experiments don't corrupt later
        # ones. Captured as strings (not paths) so an in-iteration edit to
        # strategy.yaml never disturbs the snapshot.
        self.baseline_strategy_yaml = (self.repo_root / "strategy.yaml").read_text(encoding="utf-8")
        indicators_dir = self.repo_root / "indicators"
        self.baseline_indicator_files: dict[str, str] = {
            p.name: p.read_text(encoding="utf-8") for p in sorted(indicators_dir.glob("*.py"))
        }

    # ---- public ----

    def run(
        self,
        *,
        n_iterations: int = 3,
        budget_per_iteration: int = 5,
        train_split: str = "split_A",
        val_split: str = "split_B",
        sealed_split: str = "split_C",
        report_path_template: str | None = None,
    ) -> CampaignResult:
        """Run the campaign. Returns a CampaignResult; persistence happens
        in-place under repo_root (so the user can inspect alpha_cards/ and
        memory/ after the call returns)."""

        out = CampaignResult(
            n_iterations=n_iterations,
            budget_per_iteration=budget_per_iteration,
        )
        # Per-iteration Discovery creates its own LiveStatusWriter (which
        # truncates the jsonl on init). To keep one continuous dashboard
        # across the whole campaign, we own the writer at this scope and
        # let each Discovery share it instead.
        import time as _time

        from lbg.orchestrator.live_status import LiveStatusWriter

        live = LiveStatusWriter(self.repo_root)
        t_start = _time.monotonic()

        for i in range(n_iterations):
            logger.info(
                "campaign iteration %d / %d (budget=%d)",
                i + 1,
                n_iterations,
                budget_per_iteration,
            )
            self._reset_for_iteration()
            alpha_before = self._count_alpha_cards()
            disc = Discovery(
                repo_root=self.repo_root,
                runner=self.runner,
                gate_config=self.gate_config,
            )
            disc.iter_id = i
            # Share the campaign-scoped live writer so all iterations
            # append into the same jsonl (Discovery's default writer
            # would truncate it on each iteration's __init__).
            disc.live = live
            vault_path = (
                self.repo_root / "campaigns" / f"iteration_{i:03d}" / "sealed_test_final.json"
            )
            vault_path.parent.mkdir(parents=True, exist_ok=True)
            vault = SealedVault(vault_path)
            report_path: Path | None = None
            if report_path_template is not None:
                report_path = Path(report_path_template.format(iteration=i))

            disc_result = disc.run(
                budget=budget_per_iteration,
                train_split=train_split,
                val_split=val_split,
                sealed_split=sealed_split,
                vault=vault,
                report_path=report_path,
            )
            alpha_after = self._count_alpha_cards()
            sealed_summary = self._read_sealed_summary(vault_path)
            out.iteration_results.append(
                IterationResult(
                    iteration_id=i,
                    discovery_result=disc_result,
                    alpha_cards_total_after=alpha_after,
                    alpha_cards_emitted_this_iteration=alpha_after - alpha_before,
                    sealed_summary=sealed_summary,
                )
            )

        # One summary file at the campaign root so a human can `cat` it.
        summary_path = self.repo_root / "campaigns" / "campaign_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(out.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        live.campaign_end(
            total_accepted=out.total_accepted,
            total_alpha_cards=out.total_alpha_cards,
            elapsed_sec=_time.monotonic() - t_start,
        )
        return out

    # ---- internals ----

    def _reset_for_iteration(self) -> None:
        """Restore baseline strategy + indicators; wipe event memory + runs/
        + artifacts/sealed/. Semantic memory and alpha_cards/ are preserved.
        """
        # strategy.yaml back to snapshot.
        (self.repo_root / "strategy.yaml").write_text(self.baseline_strategy_yaml, encoding="utf-8")

        # Remove any indicators added by previous iterations; restore the
        # baseline files verbatim in case an earlier iteration corrupted them.
        indicators_dir = self.repo_root / "indicators"
        indicators_dir.mkdir(exist_ok=True)
        for p in list(indicators_dir.glob("*.py")):
            if p.name not in self.baseline_indicator_files:
                p.unlink()
        for name, content in self.baseline_indicator_files.items():
            (indicators_dir / name).write_text(content, encoding="utf-8")

        # Wipe event memory but NOT semantic memory (.md) or skills/ or alpha_cards/.
        memory_dir = self.repo_root / "memory"
        if memory_dir.exists():
            for fname in _EVENT_MEMORY_FILENAMES:
                p = memory_dir / fname
                if p.exists():
                    p.unlink()

        # Wipe per-trial run artifacts.
        runs_dir = self.repo_root / "runs"
        if runs_dir.exists():
            shutil.rmtree(runs_dir)

        # Wipe the global sealed dir so each iteration's vault is fresh.
        # Per-iteration sealed files live under campaigns/iteration_NNN/ so
        # they are not affected; this only resets the legacy artifacts/sealed/.
        sealed_dir = self.repo_root / "artifacts" / "sealed"
        if sealed_dir.exists():
            shutil.rmtree(sealed_dir)

    def _count_alpha_cards(self) -> int:
        idx = self.repo_root / "alpha_cards" / "index.jsonl"
        if not idx.exists():
            return 0
        return sum(1 for line in idx.read_text(encoding="utf-8").splitlines() if line.strip())

    @staticmethod
    def _read_sealed_summary(vault_path: Path) -> dict[str, Any] | None:
        if not vault_path.exists():
            return None
        try:
            return json.loads(vault_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


__all__ = ["CampaignResult", "CampaignRunner", "IterationResult"]
