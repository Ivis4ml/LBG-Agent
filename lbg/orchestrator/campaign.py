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

import yaml

from lbg.gate import GateConfig
from lbg.git_manager import GitManager
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


def _library_cards_summary(library) -> list[dict[str, Any]]:
    """Flatten `AlphaCardLibrary.load_cards()` into JSON dicts for the
    LiveStatusWriter library_snapshot event. Keeps just the fields the
    dashboard panel renders -- omits raw indicator source / dossier blobs
    so the JSONL line stays compact.
    """
    out: list[dict[str, Any]] = []
    for card in library.load_cards():
        entry: dict[str, Any] = {
            "alpha_id": card.alpha_id,
            "source_trial": card.source_trial,
            "fn": card.signal.fn,
            "indicator": card.signal.indicator,
            "params": dict(card.signal.params),
        }
        if card.attach_config is not None:
            entry["attach_config"] = {
                "rule": card.attach_config.rule,
                "threshold": float(card.attach_config.threshold),
                "rearm_threshold": (
                    float(card.attach_config.rearm_threshold)
                    if card.attach_config.rearm_threshold is not None
                    else None
                ),
                "target": card.attach_config.target,
            }
        else:
            entry["attach_config"] = None
        sealed = card.evidence.sealed_summary
        if sealed is not None:
            # A failed per-card validation persists `{"validation_error": <str>}`
            # in sealed_summary; skip non-numeric values so the snapshot does
            # not crash on `float(<str>)`. Numeric metrics + bool flags coerce
            # fine (bool subclasses int).
            entry["sealed_summary"] = {
                k: float(v) for k, v in sealed.items() if isinstance(v, (int, float))
            }
        else:
            entry["sealed_summary"] = None
        if card.dossier_link is not None:
            entry["dossier_factor"] = card.dossier_link.factor_name
        else:
            entry["dossier_factor"] = None
        out.append(entry)
    return out


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
        library_dir: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runner = runner
        # Inject the gate per-iteration. None falls back to Discovery's strict
        # default (PROPOSAL §7). For experiment-stage campaigns that want a
        # non-zero accept rate, pass GateConfig.permissive() explicitly.
        self.gate_config = gate_config
        # Durable library destination for cards/indicator-source/dossiers. If
        # None, library sync is disabled (the run-local alpha_cards/ still
        # gets written, just not copied anywhere persistent).
        self.library_dir = Path(library_dir).resolve() if library_dir is not None else None

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
        seal_only_last_iteration: bool = False,
        carry_strategy: bool = False,
    ) -> CampaignResult:
        """Run the campaign. Returns a CampaignResult; persistence happens
        in-place under repo_root (so the user can inspect alpha_cards/ and
        memory/ after the call returns).

        seal_only_last_iteration=True opens the sealed window exactly once
        (publication mode). The default False keeps the per-iteration seal
        (development mode) so the dashboard can show H1 progress every
        iteration. Per-iteration sealing reuses the same split_C window
        across iterations; that disqualifies the sealed numbers as
        independent evidence and is documented as dev-only in the
        STAGE1_REPORT's sealed-policy note.
        """

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
            # carry_strategy=True keeps the strategy + indicators/ from
            # the prior iteration intact so iter 2+ builds on top of
            # iter 1's accepted edits -- this is what enables multi-
            # factor accumulation (PROPOSAL §17 spirit). Event memory
            # is always wiped between iters either way.
            if i == 0 or not carry_strategy:
                self._reset_for_iteration()
                # At iter 0 (and every iter when carry_strategy is off),
                # auto-inject library-validated factors into the baseline.
                # Skipped if no library is configured. Idempotent: cards
                # already present in the strategy are not re-added.
                if i == 0 and self.library_dir is not None:
                    n_injected = self._inject_library_into_baseline()
                    if n_injected:
                        logger.info(
                            "library auto-inject: wired %d validated factor(s) "
                            "into baseline strategy",
                            n_injected,
                        )
                    # Emit a snapshot before any trial runs so the
                    # dashboard's library panel populates immediately
                    # rather than waiting for the first sync.
                    try:
                        from lbg.alpha_card_library import AlphaCardLibrary

                        _lib = AlphaCardLibrary(self.library_dir)
                        target_goal = (
                            self.gate_config.library_diversity_goal
                            if self.gate_config is not None
                            and self.gate_config.library_diversity_goal > 0
                            else 5
                        )
                        live.library_snapshot(
                            iter_id=0,
                            cards=_library_cards_summary(_lib),
                            target_distinct_fns=target_goal,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("startup library_snapshot failed: %s", e)
            else:
                # Wipe only event memory + runs/ + artifacts/sealed/;
                # leave strategy.yaml and indicators/ as the prior iter
                # ended them.
                self._reset_event_memory_only()
            alpha_before = self._count_alpha_cards()
            disc = Discovery(
                repo_root=self.repo_root,
                runner=self.runner,
                gate_config=self.gate_config,
                live=live,  # share the campaign writer across iterations
                library_dir=self.library_dir,
            )
            disc.iter_id = i
            is_last_iter = i == n_iterations - 1
            seal_this_iter = (not seal_only_last_iteration) or is_last_iter
            vault: SealedVault | None
            vault_path: Path | None
            if seal_this_iter:
                vault_path = (
                    self.repo_root / "campaigns" / f"iteration_{i:03d}" / "sealed_test_final.json"
                )
                vault_path.parent.mkdir(parents=True, exist_ok=True)
                vault = SealedVault(vault_path)
            else:
                vault_path = None
                vault = None
            report_path: Path | None = None
            if report_path_template is not None:
                report_path = Path(report_path_template.format(iteration=i))

            disc_result = disc.run(
                budget=budget_per_iteration,
                train_split=train_split,
                val_split=val_split,
                sealed_split=sealed_split,
                seal_at_end=seal_this_iter,
                vault=vault,
                report_path=report_path,
            )
            alpha_after = self._count_alpha_cards()
            sealed_summary = (
                self._read_sealed_summary(vault_path) if vault_path is not None else None
            )
            # Persist newly-emitted cards into the durable library (proposal
            # §20 lock #30). Optional -- if no library_dir is configured we
            # just leave alpha_cards/ where it landed under the run root.
            if self.library_dir is not None:
                from lbg.alpha_card_library import AlphaCardLibrary
                from lbg.goal import update_after_sync as _goal_update

                lib = AlphaCardLibrary(self.library_dir)
                sync = lib.sync_from_run(self.repo_root, iteration_id=i)
                if sync.new_cards or sync.duplicate_cards or sync.new_dossiers:
                    logger.info(
                        "library sync iter %d: +%d new cards (%d dup), +%d dossiers, total %d",
                        i,
                        sync.new_cards,
                        sync.duplicate_cards,
                        sync.new_dossiers,
                        sync.library_total,
                    )
                # Emit a library snapshot to the dashboard so the
                # browser-side library panel reflects the current state
                # (factors, params, attach configs, sealed CI). Read
                # cards from disk to capture the freshly-synced state.
                try:
                    target_goal = (
                        self.gate_config.library_diversity_goal
                        if self.gate_config is not None
                        and self.gate_config.library_diversity_goal > 0
                        else 5
                    )
                    live.library_snapshot(
                        iter_id=i,
                        cards=_library_cards_summary(lib),
                        target_distinct_fns=target_goal,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("library_snapshot emit failed: %s", e)
                # Update goal_state.json + append goal_progress.jsonl in the
                # SOURCE repo (parent of library_dir), so progress carries
                # across campaigns regardless of /tmp staging. The runtime
                # repo_root is the /tmp run dir, which isn't durable.
                source_repo = self.library_dir.parent
                try:
                    goal_state = _goal_update(
                        source_repo,
                        library_dir=self.library_dir,
                        run_root=self.repo_root,
                        iteration_id=i,
                    )
                    if goal_state.met:
                        logger.info(
                            "/goal MET: %d/%d distinct factors",
                            goal_state.current_distinct_fns,
                            goal_state.target_distinct_fns,
                        )
                    elif sync.new_cards:
                        logger.info(
                            "/goal progress: %d/%d distinct factors (+%d this iter)",
                            goal_state.current_distinct_fns,
                            goal_state.target_distinct_fns,
                            sync.new_cards,
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning("goal_state update failed: %s", e)
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

    def _inject_library_into_baseline(self) -> int:
        """Read library cards and wire them into the baseline strategy.

        For each accepted card with `attach_config`:
          - copy `<library_dir>/indicators/<fn>.py` into `<repo>/indicators/`
            (skip if a same-named file already exists with identical content)
          - append an IndicatorSpec to the strategy
          - append a Filter to `strategy.filters` (target=entry) or
            `strategy.exit_filters` (target=exit) with the recorded
            threshold + optional rearm_threshold

        Returns the number of cards actually injected. Cards whose fn is
        already in the baseline are skipped (no double-wire). Cards with
        `attach_config=None` are also skipped (legacy schema, can't
        replay wiring).

        The mutation re-writes `<repo>/strategy.yaml` AND updates
        `self.baseline_strategy_yaml` so subsequent `_reset_for_iteration`
        calls in this campaign restore the *augmented* baseline.
        """
        if self.library_dir is None:
            return 0
        from lbg.alpha_card_library import AlphaCardLibrary
        from lbg.dsl import load_strategy
        from lbg.dsl.schema import (
            IndicatorAboveFilter,
            IndicatorBelowFilter,
            IndicatorSpec,
        )

        library = AlphaCardLibrary(self.library_dir)
        cards = library.load_cards()
        if not cards:
            return 0

        strategy_path = self.repo_root / "strategy.yaml"
        strategy = load_strategy(strategy_path)
        existing_fns = {s.fn for s in strategy.indicators}
        existing_names = {s.name for s in strategy.indicators}

        indicators_changes: list[IndicatorSpec] = []
        entry_filters: list = list(strategy.filters)
        exit_filters: list = list(strategy.exit_filters)
        n_injected = 0

        for card in cards:
            if card.status.value != "accepted":
                continue
            if card.attach_config is None:
                # Old cards without recorded attach. Skip rather than guess.
                logger.warning(
                    "library card %s has no attach_config; skipping auto-inject",
                    card.alpha_id,
                )
                continue
            if card.signal.fn in existing_fns or card.signal.indicator in existing_names:
                continue

            # Copy the indicator source.
            indicator_src = self.library_dir / "indicators" / f"{card.signal.fn}.py"
            if not indicator_src.exists():
                logger.warning(
                    "library card %s references missing indicator file %s; skip",
                    card.alpha_id,
                    indicator_src,
                )
                continue
            dest = self.repo_root / "indicators" / f"{card.signal.fn}.py"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(indicator_src.read_text(encoding="utf-8"), encoding="utf-8")

            # Add IndicatorSpec.
            indicators_changes.append(
                IndicatorSpec(
                    name=card.signal.indicator,
                    fn=card.signal.fn,
                    params=dict(card.signal.params),
                )
            )

            # Build the Filter from attach_config and route to entry/exit.
            attach = card.attach_config
            filter_kwargs = {
                "rule": attach.rule,
                "indicator": card.signal.indicator,
                "threshold": attach.threshold,
            }
            if attach.rearm_threshold is not None:
                filter_kwargs["rearm_threshold"] = attach.rearm_threshold
            if attach.rule == "indicator_above":
                flt = IndicatorAboveFilter(**filter_kwargs)
            elif attach.rule == "indicator_below":
                flt = IndicatorBelowFilter(**filter_kwargs)
            else:
                logger.warning(
                    "library card %s has unknown attach rule %r; skip",
                    card.alpha_id,
                    attach.rule,
                )
                continue
            if attach.target == "exit":
                exit_filters.append(flt)
            else:
                entry_filters.append(flt)

            existing_fns.add(card.signal.fn)
            existing_names.add(card.signal.indicator)
            n_injected += 1

        if n_injected == 0:
            return 0

        # Rebuild strategy with the augmented lists.
        new_strategy = strategy.model_copy(
            update={
                "indicators": list(strategy.indicators) + indicators_changes,
                "filters": entry_filters,
                "exit_filters": exit_filters,
            }
        )
        new_yaml = yaml.safe_dump(
            new_strategy.model_dump(mode="python"),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        strategy_path.write_text(new_yaml, encoding="utf-8")
        # Update the snapshot so subsequent iter resets restore the
        # AUGMENTED baseline, not the pre-inject one.
        self.baseline_strategy_yaml = new_yaml
        # Snapshot the indicators dir too -- so reset preserves the
        # newly-copied library .py files.
        indicators_dir = self.repo_root / "indicators"
        self.baseline_indicator_files = {
            p.name: p.read_text(encoding="utf-8") for p in sorted(indicators_dir.glob("*.py"))
        }
        # Commit the injected indicator files + strategy.yaml so that
        # trial commits later in the campaign have a proper parent
        # containing the library factors. Without this commit the
        # files exist on disk but are unknown to git, and sealed
        # validation's `git show <parent>:indicators/<fn>.py` fails
        # with exit 128 ("path exists on disk, but not in commit").
        # Originally surfaced by Step 4's null calibration where
        # codex rep_007's `add_indicator dispersion_regime` errored
        # during sealed CI computation; live v26 campaigns happen to
        # have worked because the first trial after auto-inject was
        # always an add_indicator that pulled the library files into
        # its commit, but that was accidental.
        try:
            git = GitManager(self.repo_root)
            git.stage(["strategy.yaml", "indicators/"])
            git.commit(f"library auto-inject: {n_injected} factor(s)", allow_empty=False)
        except Exception as e:  # noqa: BLE001
            # Auto-inject still mutated the working tree; failing to
            # commit just degrades subsequent sealed validation, not
            # the current iteration. Log loudly so the issue surfaces
            # in the dashboard / logs.
            logger.warning(
                "library auto-inject git commit failed: %s — sealed validation "
                "may fail to materialize the parent commit for the affected cards",
                e,
            )
        return n_injected

    def _reset_event_memory_only(self) -> None:
        """Subset of _reset_for_iteration: keep strategy.yaml + indicators/
        intact (carry_strategy=True path), wipe only event memory + runs/
        + the global sealed dir. Used between iters when the user wants
        each iter to build on the previous one's accepted edits.
        """
        memory_dir = self.repo_root / "memory"
        if memory_dir.exists():
            for fname in _EVENT_MEMORY_FILENAMES:
                p = memory_dir / fname
                if p.exists():
                    p.unlink()
        runs_dir = self.repo_root / "runs"
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
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
