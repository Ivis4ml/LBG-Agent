"""End-to-end Discovery loop (PROPOSAL.html §6.5 Algorithm 1).

Wires Editor → CandidateBuilder → InvariantRunner → BacktestRunner →
ValidationGate → HypothesisScorer → Reflector → MemoryManager → GitManager
into one deterministic outer loop. At the end of the budget the sealed
window is opened exactly once and the HTML report is rendered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backtest import run_backtest
from lbg.alpha_cards import (
    AlphaCardWriter,
    match_dossier_by_name,
    match_dossier_from_citation,
)
from lbg.builder import CandidateBuilder, CandidateBuildError
from lbg.data.loader import load_split
from lbg.dsl import load_strategy
from lbg.dsl.schema import Strategy
from lbg.gate import (
    GateConfig,
    TrialOutcome,
    complexity_score,
    decide,
    redact,
    score_hypothesis,
)
from lbg.git_manager import GitCommandError, GitManager
from lbg.invariants import check_prefix_stability, run_ast_checks
from lbg.memory import MemoryManager
from lbg.memory.records import (
    InvariantFailureRecord,
)
from lbg.orchestrator.context_builder import ContextBuilder
from lbg.orchestrator.curator import Curator
from lbg.orchestrator.redaction import RedactionError
from lbg.orchestrator.report_builder import ReportBuilder
from lbg.orchestrator.role_runner import RoleRunner, RoleRunnerError
from lbg.schemas import (
    AgentCompute,
    Decision,
    EditType,
    HypothesisBlock,
    RoleOutputs,
    TrainMetrics,
    TrialRecord,
)
from lbg.sealed_vault import SealedVault, SealedVaultError
from policy_interpreter import compute_positions

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    n_trials_attempted: int = 0
    n_invariant_failures: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_aborted: int = 0
    incumbent_strategy: Strategy | None = None
    final_incumbent_train_sharpe: float | None = None
    final_incumbent_val_sharpe: float | None = None
    sealed_metrics: dict | None = None
    accepted_trial_ids: list[int] = field(default_factory=list)


class Discovery:
    """The Stage-1 Discovery loop."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        runner: RoleRunner | None = None,
        gate_config: GateConfig | None = None,
        recent_trials_limit: int = 8,
        timeout_sec: float = 30.0,
        prefix_stability_n_samples: int = 12,
        prefix_stability_n_perturbations: int = 2,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runner = runner or RoleRunner()
        self.memory = MemoryManager(self.repo_root / "memory")
        self.git = GitManager(self.repo_root)
        self.builder = CandidateBuilder(self.repo_root)
        self.context_builder = ContextBuilder(
            self.memory,
            repo_root=self.repo_root,
            recent_trials_limit=recent_trials_limit,
        )
        self.curator = Curator(self.runner, self.memory, git=self.git)
        self.alpha_writer = AlphaCardWriter(self.repo_root)
        self.gate_config = gate_config or GateConfig()
        self.timeout_sec = timeout_sec
        self.prefix_n = prefix_stability_n_samples
        self.prefix_per_t = prefix_stability_n_perturbations

        self.strategy_path = self.repo_root / "strategy.yaml"
        self.indicators_dir = self.repo_root / "indicators"
        self.runs_dir = self.repo_root / "runs"

    # ---- public entry ----

    def run(
        self,
        *,
        budget: int = 5,
        train_split: str = "split_A",
        val_split: str = "split_B",
        sealed_split: str = "split_C",
        seal_at_end: bool = True,
        vault: SealedVault | None = None,
        report_path: str | Path | None = None,
    ) -> DiscoveryResult:
        df_train = load_split(train_split)  # type: ignore[arg-type]
        df_val = load_split(val_split)  # type: ignore[arg-type]

        incumbent_strategy = load_strategy(self.strategy_path)
        incumbent_outcome = self._evaluate(incumbent_strategy, df_train, df_val)
        logger.info(
            "incumbent baseline: train_sharpe=%.3f val_sharpe=%.3f complexity=%.2f",
            incumbent_outcome.train.sharpe,
            incumbent_outcome.validation.sharpe,
            incumbent_outcome.complexity,
        )

        result = DiscoveryResult(incumbent_strategy=incumbent_strategy)

        # Sequential trial_ids so each budget step gets a unique id even when
        # the Editor aborts before producing a parseable proposal. Otherwise
        # `next_trial_id()` (driven by trials.jsonl) would stay pinned to the
        # same value and the loop could waste the entire budget retrying.
        start_trial_id = self.memory.next_trial_id()
        for i in range(budget):
            trial_id = start_trial_id + i
            result.n_trials_attempted += 1
            try:
                step = self._run_one_trial(
                    trial_id=trial_id,
                    incumbent_strategy=incumbent_strategy,
                    incumbent_outcome=incumbent_outcome,
                    df_train=df_train,
                    df_val=df_val,
                )
            except (RoleRunnerError, RedactionError) as e:
                logger.warning("trial %d aborted: %s", trial_id, e)
                self._restore_working_tree(incumbent_strategy)
                result.n_aborted += 1
                continue

            if step is None:
                # Invariant or build failure -- already logged.
                result.n_invariant_failures += 1
                self._restore_working_tree(incumbent_strategy)
                continue

            if step.accepted:
                result.n_accepted += 1
                result.accepted_trial_ids.append(trial_id)
                incumbent_strategy = step.new_strategy
                incumbent_outcome = step.candidate_outcome
                logger.info("trial %d accepted: %s", trial_id, step.summary)
            else:
                result.n_rejected += 1
                self._restore_working_tree(incumbent_strategy)
                logger.info("trial %d rejected (%s): %s", trial_id, step.gate_reason, step.summary)

            # Curator (shadow) every N accepted trials.
            if self.curator.should_run(result.n_accepted):
                cycle = result.n_accepted // self.curator.cycle_interval
                try:
                    self.curator.run(cycle=cycle, accepted_count=result.n_accepted)
                    logger.info("curator cycle %d complete", cycle)
                except Exception as e:  # noqa: BLE001
                    logger.warning("curator cycle %d failed: %s", cycle, e)

        result.incumbent_strategy = incumbent_strategy
        result.final_incumbent_train_sharpe = incumbent_outcome.train.sharpe
        result.final_incumbent_val_sharpe = incumbent_outcome.validation.sharpe

        if seal_at_end:
            sealed_metrics = self._seal(incumbent_strategy, sealed_split, vault)
            result.sealed_metrics = sealed_metrics

        if report_path is not None:
            ReportBuilder(self.memory, vault, output_path=report_path).build()

        return result

    # ---- internals ----

    @dataclass(frozen=True)
    class _StepResult:
        accepted: bool
        new_strategy: Strategy
        candidate_outcome: TrialOutcome
        gate_reason: str
        summary: str

    def _run_one_trial(
        self,
        *,
        trial_id: int,
        incumbent_strategy: Strategy,
        incumbent_outcome: TrialOutcome,
        df_train,
        df_val,
    ):
        ctx = self.context_builder.editor_view(incumbent_strategy)
        editor_result = self.runner.editor(ctx, trial_id=trial_id)

        # Apply the parsed edit.
        try:
            apply_result = self.builder.apply(
                incumbent_strategy, editor_result.proposal, editor_result.payload
            )
        except (CandidateBuildError, NotImplementedError) as e:
            self._record_invariant_failure(
                trial_id,
                "candidate_build",
                str(e),
                "<builder>",
            )
            return None

        # Static AST checks on any new .py file the edit produced.
        ast_failures = []
        for path in apply_result.touched_paths:
            if path.suffix == ".py":
                ast_failures.extend(run_ast_checks(path))
        for v in ast_failures:
            self._record_invariant_failure(trial_id, v.name, v.message, v.file, v.line)
        if ast_failures:
            return None

        # Prefix-stability on the candidate's full strategy, by running each
        # candidate-added indicator on real train data.
        prefix_failures = []
        for path in apply_result.touched_paths:
            if path.suffix == ".py":
                try:
                    from lbg.sandbox import load_indicator

                    fn = load_indicator(path)
                    params = next(
                        (
                            i.params
                            for i in apply_result.new_strategy.indicators
                            if i.fn == path.stem
                        ),
                        {},
                    )
                    prefix_failures.extend(
                        check_prefix_stability(
                            fn,
                            df_train,
                            params=params,
                            n_samples=self.prefix_n,
                            n_perturbations_per_t=self.prefix_per_t,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    self._record_invariant_failure(trial_id, "sandbox_load", str(e), str(path))
                    return None
        for v in prefix_failures:
            self._record_invariant_failure(trial_id, v.name, v.message, v.file, v.line)
        if prefix_failures:
            return None

        # Backtests + gate.
        try:
            candidate_outcome = self._evaluate(apply_result.new_strategy, df_train, df_val)
        except Exception as e:  # noqa: BLE001
            self._record_invariant_failure(trial_id, "backtest_error", str(e), "<backtest>")
            return None

        gate_decision = decide(incumbent_outcome, candidate_outcome, self.gate_config)
        signal = redact(gate_decision)

        delta_sharpe = candidate_outcome.train.sharpe - incumbent_outcome.train.sharpe
        hyp_outcome = score_hypothesis(
            expected_train=editor_result.proposal.expected_train_signal,
            expected_validation=editor_result.proposal.expected_validation_signal,
            delta_sharpe_train=delta_sharpe,
            actual_validation=signal,
        )

        # Reflector.
        refl_result = self.runner.reflector(
            trial_id=trial_id,
            editor_context=ctx,
            proposal=editor_result.proposal,
            hypothesis_outcome=hyp_outcome,
            actual_validation_signal=signal,
            actual_train_metrics=candidate_outcome.train,
        )

        # Persist run artifacts.
        run_dir = self.runs_dir / f"{trial_id:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        editor_yaml_path = run_dir / "editor.yaml"
        reflector_yaml_path = run_dir / "reflector.yaml"
        editor_yaml_path.write_text(editor_result.raw_text, encoding="utf-8")
        reflector_yaml_path.write_text(refl_result.raw_text, encoding="utf-8")

        # Memory writes (jsonl + md).
        try:
            parent_commit = self.git.head_sha()
        except GitCommandError:
            parent_commit = "0" * 40
        record = TrialRecord(
            trial_id=trial_id,
            parent_commit=parent_commit,
            candidate_commit="_pending_",
            role_outputs=RoleOutputs(
                editor_output_path=str(editor_yaml_path.relative_to(self.repo_root)),
                reflector_output_path=str(reflector_yaml_path.relative_to(self.repo_root)),
            ),
            edit=apply_result.edit_summary,
            hypothesis=HypothesisBlock(
                text=editor_result.proposal.hypothesis,
                expected_train_signal=editor_result.proposal.expected_train_signal,
                expected_validation_signal=editor_result.proposal.expected_validation_signal,
            ),
            invariants={"ast_static": "pass", "prefix_stability": "pass"},
            train_metrics=TrainMetrics(
                sharpe=candidate_outcome.train.sharpe,
                max_drawdown=candidate_outcome.train.max_drawdown,
                turnover=candidate_outcome.train.turnover,
                num_trades=candidate_outcome.train.num_trades,
            ),
            validation_signal=signal,
            hypothesis_outcome=hyp_outcome,
            decision=Decision.ACCEPT if gate_decision.accepted else Decision.REJECT,
            complexity_before=incumbent_outcome.complexity,
            complexity_after=candidate_outcome.complexity,
            agent_compute=AgentCompute(
                editor_input_tokens=editor_result.compute.input_tokens,
                editor_output_tokens=editor_result.compute.output_tokens,
                reflector_input_tokens=refl_result.compute.input_tokens,
                reflector_output_tokens=refl_result.compute.output_tokens,
                model_editor=editor_result.compute.model,
                model_reflector=refl_result.compute.model,
                wall_clock_sec=editor_result.compute.wall_clock_sec
                + refl_result.compute.wall_clock_sec,
            ),
            fallback_if_rejected=editor_result.proposal.fallback_if_rejected,
            cited_factors=list(editor_result.proposal.cited_factors),
        )
        self.memory.append_trial(record)
        self.memory.append_reflection(refl_result.record)
        self.memory.append_agent_compute(editor_result.compute)
        self.memory.append_agent_compute(refl_result.compute)
        self.memory.append_to_md("accepted_rules.md", refl_result.record.accepted_rules_updates)
        self.memory.append_to_md(
            "failed_directions.md", refl_result.record.failed_directions_updates
        )
        self.memory.append_to_md("open_questions.md", refl_result.record.open_questions_updates)
        self.memory.append_to_md("do_not_repeat.md", refl_result.record.do_not_repeat_updates)

        # Alpha card: emit on accepted add_indicator trials only (PROPOSAL §6.6).
        # H1 (PROPOSAL §4) counts cards whose sealed_summary clears the CI bar;
        # the per-trial card pins the artifact at the moment of acceptance.
        alpha_card_path: Path | None = None
        if record.decision == Decision.ACCEPT and record.edit.type == EditType.ADD_INDICATOR:
            change = editor_result.proposal.proposed_edit.change
            indicator_name = getattr(change, "name", None)
            if indicator_name:
                try:
                    # Prefer explicit Editor citation over substring guess;
                    # falls back to name-match when the Editor didn't cite.
                    dossier_link = match_dossier_from_citation(
                        editor_result.proposal.cited_factors
                    ) or match_dossier_by_name(getattr(change, "fn", indicator_name))
                    alpha_card_path = self.alpha_writer.write_for_added_indicator(
                        trial_id=trial_id,
                        source_commit=parent_commit,
                        added_indicator_name=indicator_name,
                        strategy=apply_result.new_strategy,
                        train_metrics=record.train_metrics,
                        validation_signal=record.validation_signal,
                        hypothesis_outcome=record.hypothesis_outcome,
                        dossier_link=dossier_link,
                    )
                except (ValueError, OSError) as e:
                    logger.warning("trial %d alpha card emit failed: %s", trial_id, e)

        # Git commit.
        files = [
            *(str(p.relative_to(self.repo_root)) for p in apply_result.touched_paths),
            str(self.memory.trials_path.relative_to(self.repo_root)),
            str(self.memory.reflections_path.relative_to(self.repo_root)),
            str(self.memory.agent_compute_path.relative_to(self.repo_root)),
            str(editor_yaml_path.relative_to(self.repo_root)),
            str(reflector_yaml_path.relative_to(self.repo_root)),
        ]
        if alpha_card_path is not None:
            files.append(str(alpha_card_path.relative_to(self.repo_root)))
            files.append(str(self.alpha_writer.index_path.relative_to(self.repo_root)))
        for md_name in (
            "accepted_rules.md",
            "failed_directions.md",
            "open_questions.md",
            "do_not_repeat.md",
        ):
            p = self.memory.semantic_memory_path(md_name)
            if p.exists():
                files.append(str(p.relative_to(self.repo_root)))
        try:
            self.git.stage([f for f in files if (self.repo_root / f).exists()])
            self.git.commit_trial(trial_id, apply_result.edit_summary.summary)
        except (GitCommandError, ValueError) as e:
            logger.warning("trial %d git commit failed: %s", trial_id, e)

        return self._StepResult(
            accepted=gate_decision.accepted,
            new_strategy=apply_result.new_strategy,
            candidate_outcome=candidate_outcome,
            gate_reason=gate_decision.reason,
            summary=apply_result.edit_summary.summary,
        )

    def _evaluate(self, strategy: Strategy, df_train, df_val) -> TrialOutcome:
        positions_t = compute_positions(
            strategy, df_train, indicators_dir=self.indicators_dir, timeout_sec=self.timeout_sec
        )
        positions_v = compute_positions(
            strategy, df_val, indicators_dir=self.indicators_dir, timeout_sec=self.timeout_sec
        )
        return TrialOutcome(
            train=run_backtest(positions_t, df_train),
            validation=run_backtest(positions_v, df_val),
            complexity=complexity_score(strategy, indicators_dir=self.indicators_dir),
        )

    def _restore_working_tree(self, incumbent: Strategy) -> None:
        """Rewrite strategy.yaml back to `incumbent` and drop untracked
        indicator files. Idempotent."""
        import yaml as _yaml

        text = _yaml.safe_dump(
            incumbent.model_dump(mode="python"),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        self.strategy_path.write_text(text, encoding="utf-8")
        # Drop indicator files that aren't referenced by `incumbent`.
        referenced = {i.fn for i in incumbent.indicators}
        if self.indicators_dir.exists():
            for py in self.indicators_dir.glob("*.py"):
                if py.stem not in referenced and py.stem != "__init__":
                    py.unlink()

    def _record_invariant_failure(
        self,
        trial_id: int,
        name: str,
        message: str,
        file: str,
        line: int = 0,
    ) -> None:
        try:
            parent = self.git.head_sha()
        except GitCommandError:
            parent = "0" * 40
        self.memory.append_invariant_failure(
            InvariantFailureRecord(
                trial_id=trial_id,
                parent_commit=parent,
                invariant_name=name,
                message=message,
                file=file,
                line=line,
            )
        )

    def _seal(
        self,
        incumbent: Strategy,
        sealed_split: str,
        vault: SealedVault | None,
        *,
        analysis_plan=None,
    ) -> dict:
        from lbg.verdict import DEFAULT_ANALYSIS_PLAN, compute_h1_verdict

        plan = analysis_plan or DEFAULT_ANALYSIS_PLAN

        df_sealed = load_split(sealed_split)  # type: ignore[arg-type]
        positions = compute_positions(
            incumbent, df_sealed, indicators_dir=self.indicators_dir, timeout_sec=self.timeout_sec
        )
        sealed = run_backtest(positions, df_sealed)

        # H1 verdict: pre-registered moving block bootstrap vs best baseline.
        try:
            verdict = compute_h1_verdict(sealed.returns, df_sealed, plan=plan)
            h1 = verdict.to_dict()
        except Exception as e:  # noqa: BLE001
            logger.warning("H1 verdict computation failed: %s", e)
            h1 = {"error": str(e)}

        payload = {
            "strategy_name": incumbent.name,
            "n_bars_used": sealed.n_bars_used,
            "sharpe": sealed.sharpe,
            "max_drawdown": sealed.max_drawdown,
            "turnover": sealed.turnover,
            "num_trades": sealed.num_trades,
            "cagr": sealed.cagr,
            "final_equity": sealed.final_equity,
            "complexity": complexity_score(incumbent, indicators_dir=self.indicators_dir),
            "h1_verdict": h1,
        }
        if vault is not None:
            try:
                vault.open_and_write(payload)
                logger.info("sealed vault written to %s", vault.path)
            except SealedVaultError as e:
                logger.warning("sealed vault already opened: %s", e)
        return payload
