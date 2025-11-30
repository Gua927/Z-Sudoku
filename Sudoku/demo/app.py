from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr

from sampler import SudokuDiffusionSampler, build_sampler

CUSTOM_CSS = """
:root {
  --board-border: #1e1e2f;
  --given-bg: #f0f4ff;
  --cell-border: #c4c4d4;
  --error-bg: #ffd9d9;
  --accent: #4c7eff;
}

.board {
  border: 2px solid var(--board-border);
  border-radius: 8px;
  padding: 0.6rem;
  background: #fff;
  box-shadow: 0 4px 16px rgba(0,0,0,0.07);
}

.board-title {
  font-weight: 600;
  margin-bottom: 0.4rem;
  color: var(--board-border);
}

.board table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
}

.board td {
  border: 1px solid var(--cell-border);
  width: 11%;
  height: 32px;
  text-align: center;
  font-size: 1.05rem;
  font-weight: 600;
  color: #1a1a1a;
}

.board td.given {
  background: var(--given-bg);
}

.board td.error {
  background: var(--error-bg);
  color: #b00020;
}

.board td:nth-child(3n) {
  border-right: 2px solid var(--board-border);
}

.board tr:nth-child(3n) td {
  border-bottom: 2px solid var(--board-border);
}

.history-panel {
    margin-top: 0.8rem;
    max-height: 420px;
    overflow-y: auto;
}

.history-step {
    border: 1px solid var(--cell-border);
    border-radius: 6px;
    padding: 0.45rem 0.6rem;
    margin-bottom: 0.6rem;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.history-step-title {
    font-weight: 600;
    color: var(--board-border);
    margin-bottom: 0.25rem;
}

.history-step pre {
    margin: 0;
    background: #f7f8ff;
    padding: 0.4rem;
    border-radius: 4px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.85rem;
    line-height: 1.2;
    white-space: pre;
}
"""


def normalize_puzzle(raw: str) -> str:
    digits: List[str] = []
    for ch in raw.strip():
        if ch in "123456789":
            digits.append(ch)
        elif ch in "0.":
            digits.append("0")
    if len(digits) != 81:
        raise ValueError("请提供81个字符的题目串（用0或.表示空格）。")
    return "".join(digits)


def split_rows(values: str) -> List[str]:
    return [values[i * 9 : (i + 1) * 9] for i in range(9)]


def find_given_conflicts(puzzle: str, solution: str) -> List[int]:
    return [idx for idx, (p, s) in enumerate(zip(puzzle, solution)) if p != "0" and p != s]


def find_sudoku_conflicts(solution: str) -> List[int]:
    conflicts: set[int] = set()

    def mark(indices: Sequence[int]):
        seen: Dict[str, int] = {}
        for idx in indices:
            val = solution[idx]
            if val == "0":
                conflicts.add(idx)
                continue
            if val in seen:
                conflicts.add(idx)
                conflicts.add(seen[val])
            else:
                seen[val] = idx

    for r in range(9):
        row = [r * 9 + c for c in range(9)]
        mark(row)
    for c in range(9):
        col = [r * 9 + c for r in range(9)]
        mark(col)
    for br in range(3):
        for bc in range(3):
            box = [
                (br * 3 + dr) * 9 + (bc * 3 + dc)
                for dr in range(3)
                for dc in range(3)
            ]
            mark(box)
    return sorted(conflicts)


def render_board_html(board: str, title: str, givens: Optional[str] = None, highlights: Optional[Sequence[int]] = None) -> str:
    rows = split_rows(board)
    highlight_set = set(highlights or [])
    cells = []
    for r, row in enumerate(rows):
        row_cells = []
        for c, value in enumerate(row):
            idx = r * 9 + c
            classes: List[str] = []
            if givens and givens[idx] != "0":
                classes.append("given")
            if idx in highlight_set:
                classes.append("error")
            display = value if value != "0" else "&nbsp;"
            class_attr = " ".join(classes).strip()
            if class_attr:
                row_cells.append(f"<td class='{class_attr}'>{display}</td>")
            else:
                row_cells.append(f"<td>{display}</td>")
        cells.append(f"<tr>{''.join(row_cells)}</tr>")
    table = "".join(cells)
    return f"<div class='board'><div class='board-title'>{title}</div><table>{table}</table></div>"


def format_mask_schedule(history: List[Tuple[int, int]]) -> str:
    if not history:
        return ""
    ordered = sorted(history, key=lambda x: x[0])
    counts = " → ".join(str(item[1]) for item in ordered)
    return f"Mask progression（按步数从小到大）: {counts}"


def render_history_html(timeline: List[Dict[str, object]]) -> str:
    if not timeline:
        return "<p class='history-step-title'>暂无去噪记录。</p>"
    blocks: List[str] = []
    for entry in timeline:
        step = entry.get("step")
        masks = entry.get("remaining_masks")
        candidate = html.escape(str(entry.get("candidate", "")))
        blocks.append(
            """
            <div class="history-step">
              <div class="history-step-title">t = {step} · 剩余掩码 = {masks}</div>
              <pre>{grid}</pre>
            </div>
            """.format(step=step, masks=masks, grid=candidate)
        )
    return f"<div class='history-panel'>{''.join(blocks)}</div>"


def load_examples(csv_path: Path, limit: int) -> Dict[str, str]:
    examples: Dict[str, str] = {}
    if not csv_path.exists():
        return examples
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            examples[f"示例 #{idx}"] = row["quizzes"]
            if idx >= limit:
                break
    return examples


def build_status_markdown(solution: str, puzzle: str, elapsed: float, confidence: Optional[float], history: List[Tuple[int, int]]) -> str:
    givens_bad = find_given_conflicts(puzzle, solution)
    sudoku_bad = find_sudoku_conflicts(solution)
    valid = not sudoku_bad and all(ch in "123456789" for ch in solution)
    md_lines = [f"**推理耗时:** {elapsed:.2f}s"]
    if confidence is not None:
        md_lines.append(f"**平均置信度:** {confidence:.3f}")
    md_lines.append("**满足题目给定:** ✅" if not givens_bad else f"**满足题目给定:** ❌（冲突 {len(givens_bad)} 个)")
    if valid:
        md_lines.append("**数独合法:** ✅")
    else:
        md_lines.append("**数独合法:** ❌")
    schedule_text = format_mask_schedule(history)
    if schedule_text:
        md_lines.append(schedule_text)
    return "\n".join(md_lines)


def solve_handler(
    sampler: SudokuDiffusionSampler,
    puzzle_text: str,
    strategy: str,
    schedule: str,
    steps: int,
    temperature: float,
    seed_value: Optional[int],
) -> Tuple[str, str, str, str]:
    try:
        puzzle = normalize_puzzle(puzzle_text)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    seed = seed_value if seed_value and seed_value > 0 else None
    result = sampler.solve(
        puzzle=puzzle,
        diffusion_steps=steps,
        temperature=temperature,
        denoise_method=strategy,
        decoding_strategy=schedule,
        seed=seed,
    )
    solution = result["solution"]
    given_conflicts = find_given_conflicts(puzzle, solution)
    sudoku_conflicts = find_sudoku_conflicts(solution)
    merged_conflicts = sorted(set(given_conflicts) | set(sudoku_conflicts))
    puzzle_html = render_board_html(puzzle, "题目", givens=puzzle)
    solution_html = render_board_html(solution, "模型输出", givens=puzzle, highlights=merged_conflicts)
    status_md = build_status_markdown(solution, puzzle, result["elapsed"], result["confidence"], result["mask_history"])
    history_html = render_history_html(result.get("timeline", []))
    return solution, puzzle_html, solution_html, status_md, history_html


def build_interface(sampler: SudokuDiffusionSampler, default_puzzle: str, examples: Dict[str, str]):
    example_labels = list(examples.keys())
    default_example = example_labels[0] if example_labels else None

    with gr.Blocks(css=CUSTOM_CSS, title="Sudoku Diffusion Demo") as demo:
        gr.Markdown("""
        ## Sudoku Diffusion Demo
        使用离散扩散模型的去噪过程完成数独求解。可切换解码策略、扩散步数与温度，体验不同的推理行为。
        """)
        with gr.Row():
            with gr.Column():
                puzzle_box = gr.Textbox(
                    label="数独题目（81字符，0或. 表示空）",
                    value=default_puzzle,
                    lines=3,
                    show_copy_button=True,
                )
                if example_labels:
                    example_dropdown = gr.Dropdown(
                        choices=example_labels,
                        value=default_example,
                        label="示例题目",
                    )

                    def _load_example(choice: str) -> str:
                        return examples.get(choice, default_puzzle)

                    example_dropdown.change(_load_example, inputs=example_dropdown, outputs=puzzle_box)
            with gr.Column():
                strategy = gr.Dropdown(
                    label="解码策略",
                    choices=["origin", "topk", "topk-margin", "downk", "downk-margin"],
                    value="downk-margin",
                )
                schedule = gr.Dropdown(
                    label="调度方式",
                    choices=["linear", "cosine"],
                    value="linear",
                )
                steps = gr.Slider(5, 30, value=20, step=1, label="扩散步数")
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="温度")
                seed = gr.Number(value=114, label="随机种子 (<=0 表示随机)", precision=0)
                solve_button = gr.Button("开始推理", variant="primary")
        solution_box = gr.Textbox(label="预测解 (81字符)", interactive=False, show_copy_button=True)
        puzzle_html = gr.HTML()
        solution_html = gr.HTML()
        status_md = gr.Markdown()
        with gr.Accordion("完整去噪过程", open=False):
            history_panel = gr.HTML()

        solve_button.click(
            fn=lambda puzzle_text, strat, sched, step_count, temp, seed_val: solve_handler(
                sampler, puzzle_text, strat, sched, int(step_count), float(temp), None if seed_val is None else int(seed_val)
            ),
            inputs=[puzzle_box, strategy, schedule, steps, temperature, seed],
            outputs=[solution_box, puzzle_html, solution_html, status_md, history_panel],
        )

    return demo


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sudoku diffusion demo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "output/sudoku/Z-mask-RE/Weight2-downk0.3-margin-all-z-seed114",
        help="Checkpoint directory to load.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=repo_root / "model_config_tiny",
        help="Backbone config directory (defaults to model_config_tiny).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing.",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--examples", type=int, default=6, help="Number of sample puzzles to load.")
    return parser.parse_args()


def main():
    args = parse_args()
    sampler = build_sampler(args.checkpoint, args.base_model)
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "data/sudoku_test.csv"
    examples = load_examples(dataset_path, args.examples)
    if examples:
        first_key = next(iter(examples))
        default_puzzle = examples[first_key]
    else:
        default_puzzle = "0" * 81
    demo = build_interface(sampler, default_puzzle, examples)
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
