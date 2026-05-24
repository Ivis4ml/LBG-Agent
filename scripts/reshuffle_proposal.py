"""Re-order sections in PROPOSAL.html and PROPOSAL_ZH.html:

1. Move §3.5 (id=hlfit, "HL properties realised") to end of Part II,
   renumbered as §10.6. All forward references in §3.5 then become
   backward references.

2. Swap §6.7 (id=trialwalk, "one trial end-to-end") and §6.8
   (id=factorlib, "factor library as input"). Worked example should
   come after the input resource it uses, not before.

Anchors (id="...") are preserved so existing links still work.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_section_block(content: str, sec_id: str) -> tuple[str, int]:
    """Return (block_text, start_index) for <section id="sec_id"> through </section>."""
    pattern = rf'<section id="{sec_id}">.*?\n</section>\n'
    m = re.search(pattern, content, re.DOTALL)
    if not m:
        raise ValueError(f"section id={sec_id!r} not found")
    return m.group(0), m.start()


def move_hlfit_to_10_6(content: str) -> str:
    block, _ = find_section_block(content, "hlfit")

    # Renumber inside the block
    new_block = block.replace(
        '<span class="section-num">§ 3.5</span>',
        '<span class="section-num">§ 10.6</span>',
    )
    # Self-references in the body
    new_block = new_block.replace("§3.5 is the audit trail", "§10.6 is the audit trail")
    new_block = new_block.replace("§3.5 就是审计线索", "§10.6 就是审计线索")

    # Remove from old location
    content = content.replace(block, "", 1)

    # Insert before Part III part-marker (right after evalmath's </section>)
    target = '<div class="part-marker">\n  <span class="roman">III.</span>'
    if target not in content:
        raise RuntimeError("Part III part-marker not found")
    content = content.replace(target, new_block + "\n" + target, 1)
    return content


def swap_67_68(content: str) -> str:
    trialwalk_block, _ = find_section_block(content, "trialwalk")
    factorlib_block, _ = find_section_block(content, "factorlib")

    # Renumber: trialwalk becomes §6.8, factorlib becomes §6.7
    trialwalk_new = trialwalk_block.replace(
        '<span class="section-num">§ 6.7</span>',
        '<span class="section-num">§ 6.8</span>',
    )
    factorlib_new = factorlib_block.replace(
        '<span class="section-num">§ 6.8</span>',
        '<span class="section-num">§ 6.7</span>',
    )

    # The two sections are adjacent (trialwalk then factorlib).
    # Swap by replacing the concatenation.
    combined_old = trialwalk_block + "\n" + factorlib_block
    combined_new = factorlib_new + "\n" + trialwalk_new
    if combined_old not in content:
        # try without intervening newline
        combined_old = trialwalk_block + factorlib_block
        combined_new = factorlib_new + trialwalk_new
        if combined_old not in content:
            raise RuntimeError("trialwalk and factorlib not adjacent in expected form")
    content = content.replace(combined_old, combined_new, 1)
    return content


def update_toc_en(content: str) -> str:
    # Drop §3.5 entry from Part I
    content = content.replace(
        '  <a href="#hlfit">§3.5 HL properties realised</a>\n',
        "",
    )
    # Add §10.6 entry to end of Part II (right before Part III toc-label)
    content = content.replace(
        '  <a href="#evalmath">§10.5 Eval math</a>\n\n  <div class="toc-label">Part III · Lifecycle</div>',
        '  <a href="#evalmath">§10.5 Eval math</a>\n  <a href="#hlfit">§10.6 HL properties realised</a>\n\n  <div class="toc-label">Part III · Lifecycle</div>',
        1,
    )
    # Swap §6.7 / §6.8 entries
    content = content.replace(
        '  <a href="#trialwalk">§6.7 One trial, end to end</a>\n  <a href="#factorlib">§6.8 Factor library as input</a>',
        '  <a href="#factorlib">§6.7 Factor library as input</a>\n  <a href="#trialwalk">§6.8 One trial, end to end</a>',
        1,
    )
    return content


def update_toc_zh(content: str) -> str:
    content = content.replace(
        '  <a href="#hlfit">§3.5 兑现 HL 性质</a>\n',
        "",
    )
    content = content.replace(
        '  <a href="#evalmath">§10.5 评测数学</a>\n\n  <div class="toc-label">第三部分 · 生命周期</div>',
        '  <a href="#evalmath">§10.5 评测数学</a>\n  <a href="#hlfit">§10.6 兑现 HL 性质</a>\n\n  <div class="toc-label">第三部分 · 生命周期</div>',
        1,
    )
    content = content.replace(
        '  <a href="#trialwalk">§6.7 一次试验端到端</a>\n  <a href="#factorlib">§6.8 因子库作为原材料</a>',
        '  <a href="#factorlib">§6.7 因子库作为原材料</a>\n  <a href="#trialwalk">§6.8 一次试验端到端</a>',
        1,
    )
    return content


def process(path: Path, toc_updater) -> None:
    content = path.read_text(encoding="utf-8")
    content = move_hlfit_to_10_6(content)
    content = swap_67_68(content)
    content = toc_updater(content)
    path.write_text(content, encoding="utf-8")
    print(f"reshuffled {path}")


def main() -> int:
    process(Path("PROPOSAL.html"), update_toc_en)
    process(Path("PROPOSAL_ZH.html"), update_toc_zh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
