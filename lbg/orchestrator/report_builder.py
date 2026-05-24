"""ReportBuilder: render a static HTML report from logged artifacts.

PROPOSAL.html §6.5 line 1269. Reads `memory/trials.jsonl`, the four
semantic-memory documents, agent_compute records, and (if sealed) the
SealedVault contents, and renders one HTML file with:

  - Summary line: total trials, accepted, validation-signal histogram
  - Per-trial table: trial_id, edit_type, edit_summary, train metrics,
    validation signal, hypothesis outcome, decision
  - Semantic memory dump
  - Sealed verdict (when present)
  - Agent compute totals
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from lbg.memory import SEMANTIC_MEMORY_FILES, MemoryManager
from lbg.sealed_vault import SealedVault, SealedVaultError

DEFAULT_REPORT_PATH = Path("artifacts/reports/discovery_report.html")


class ReportBuilder:
    def __init__(
        self,
        memory: MemoryManager,
        vault: SealedVault | None = None,
        *,
        output_path: str | Path = DEFAULT_REPORT_PATH,
    ) -> None:
        self.memory = memory
        self.vault = vault
        self.output_path = Path(output_path)

    def build(self) -> Path:
        trials = self.memory.read_trials()
        reflections = self.memory.read_reflections()
        computes = self.memory.read_agent_compute()

        signals = Counter(t.validation_signal.value for t in trials)
        outcomes = Counter(t.hypothesis_outcome.value for t in trials)
        decisions = Counter(t.decision.value for t in trials)
        edit_types = Counter(t.edit.type.value for t in trials)

        input_tokens = sum(c.input_tokens for c in computes)
        output_tokens = sum(c.output_tokens for c in computes)
        wall_clock_total = sum(c.wall_clock_sec for c in computes)
        compute_by_role = Counter(c.role for c in computes)

        sealed_html = self._sealed_section()
        memory_html = self._memory_section()
        trials_html = self._trials_section(trials, reflections)

        html_doc = _TEMPLATE.format(
            n_trials=len(trials),
            n_accepted=decisions.get("accept", 0),
            n_rejected=decisions.get("reject", 0),
            signals_html=_render_counter(signals),
            outcomes_html=_render_counter(outcomes),
            edit_types_html=_render_counter(edit_types),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_clock_total=f"{wall_clock_total:.1f}",
            compute_by_role_html=_render_counter(compute_by_role),
            trials_html=trials_html,
            memory_html=memory_html,
            sealed_html=sealed_html,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html_doc, encoding="utf-8")
        return self.output_path

    def _sealed_section(self) -> str:
        if self.vault is None:
            return "<p class='dim'>No SealedVault attached.</p>"
        try:
            data = self.vault.read()
        except SealedVaultError:
            return "<p class='dim'>SealedVault not yet sealed.</p>"
        rows = "\n".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in data.items()
        )
        return f"<table class='kv'>{rows}</table>"

    def _memory_section(self) -> str:
        parts = []
        for name in SEMANTIC_MEMORY_FILES:
            content = self.memory.read_md(name).strip()
            body = html.escape(content) if content else "<em class='dim'>(empty)</em>"
            parts.append(f"<h3>{name}</h3><pre>{body}</pre>")
        return "\n".join(parts)

    def _trials_section(self, trials, reflections) -> str:
        if not trials:
            return "<p class='dim'>No trials recorded.</p>"
        explanations = {r.trial_id: r.explanation for r in reflections}
        rows = []
        for t in trials:
            metrics = (
                f"S={t.train_metrics.sharpe:.2f}, "
                f"DD={t.train_metrics.max_drawdown:.2f}, "
                f"TO={t.train_metrics.turnover:.1f}, "
                f"N={t.train_metrics.num_trades}"
            )
            explanation = explanations.get(t.trial_id, "")
            rows.append(
                "<tr>"
                f"<td>{t.trial_id}</td>"
                f"<td><code>{html.escape(t.edit.type.value)}</code></td>"
                f"<td>{html.escape(t.edit.summary)}</td>"
                f"<td><code>{metrics}</code></td>"
                f"<td>{t.validation_signal.value}</td>"
                f"<td>{t.hypothesis_outcome.value}</td>"
                f"<td>{t.decision.value}</td>"
                f"<td>{html.escape(explanation)}</td>"
                "</tr>"
            )
        body = "\n".join(rows)
        return (
            "<table class='trials'><thead><tr>"
            "<th>id</th><th>edit type</th><th>summary</th>"
            "<th>train metrics</th><th>val signal</th>"
            "<th>hypothesis outcome</th><th>decision</th>"
            "<th>reflector explanation</th>"
            "</tr></thead><tbody>"
            f"{body}"
            "</tbody></table>"
        )


def _render_counter(c: Counter) -> str:
    if not c:
        return "<em class='dim'>(none)</em>"
    items = ", ".join(f"{html.escape(k)}={v}" for k, v in c.most_common())
    return items


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LBG-Trader Discovery Report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1, h2, h3 {{ color: #111; }}
  .dim {{ color: #888; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{ background: #f4f4f6; }}
  table.kv td:first-child {{ font-weight: 600; background: #f9f9fb; width: 220px; }}
  pre {{ background: #f7f7f9; border: 1px solid #e8e8ec; padding: 10px; font-size: 12px; overflow-x: auto; }}
  code {{ font-family: SF Mono, Menlo, monospace; font-size: 12px; }}
  .summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .summary > div {{ padding: 1rem; border: 1px solid #e0e0e4; border-radius: 4px; background: #fafbfc; }}
</style>
</head>
<body>
<h1>LBG-Trader · Discovery Report</h1>

<h2>Summary</h2>
<div class="summary">
  <div>
    <strong>Total trials:</strong> {n_trials}<br>
    <strong>Accepted:</strong> {n_accepted}<br>
    <strong>Rejected:</strong> {n_rejected}
  </div>
  <div>
    <strong>Input tokens:</strong> {input_tokens}<br>
    <strong>Output tokens:</strong> {output_tokens}<br>
    <strong>LLM wall-clock total:</strong> {wall_clock_total} s
  </div>
</div>

<h3>Validation signal histogram</h3>
<p>{signals_html}</p>

<h3>Hypothesis outcome histogram</h3>
<p>{outcomes_html}</p>

<h3>Edit type histogram</h3>
<p>{edit_types_html}</p>

<h3>Compute by role</h3>
<p>{compute_by_role_html}</p>

<h2>Sealed test verdict</h2>
{sealed_html}

<h2>Per-trial</h2>
{trials_html}

<h2>Semantic memory</h2>
{memory_html}

</body>
</html>
"""
