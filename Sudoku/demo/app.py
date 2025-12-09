# nohup python demo/app.py > demo/demo_app.out &
# disown

import sys
import os
import torch
import gradio as gr
import math
import numpy as np
import time
import random
import csv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from llmtuner.tuner.mdm.trainer import CustomDiffusionTrainer

# Global model and tokenizer
model = None
tokenizer = None
diffusion_args_global = None

def load_the_model():
    global model, tokenizer, diffusion_args_global
    if model is not None:
        return

    print("Loading model...")
    model_path = "/root/code/EasyMDM/Sudoku/output/sudoku/Z-mask-RE/Weight2-downk0.3-margin-all-z-seed114"
    
    parser = HfArgumentParser((ModelArguments, FinetuningArguments, DiffusionArguments))
    model_args, finetuning_args, diffusion_args = parser.parse_args_into_dataclasses(args=[
        "--model_name_or_path", "model_config_tiny",
        "--checkpoint_dir", model_path,
        "--finetuning_type", "full",
        "--diffusion_steps", "20", 
        "--stage", "mdm",
    ])
    
    # Set defaults for demo
    diffusion_args.reunmask = True
    diffusion_args.downk_ratio = 0.3
    diffusion_args_global = diffusion_args

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, is_trainable=False, diffusion_args=diffusion_args)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print("Model loaded successfully.")

class DemoTrainer(CustomDiffusionTrainer):
    def generate_samples_generator(self, inputs, step_generated, sleep_time=0.5):
        self.model.cuda()
        self.model.eval()
        
        x = inputs['input_ids'].cuda()
        src_mask = inputs['src_mask'].bool().cuda()
        attention_mask = torch.ones_like(x) 
        batch_size = x.size(0)

        init_maskable_mask = maskable_mask = ~src_mask
        denoise_method = getattr(self.diff_args, "denoise_method", "origin")
        decoding_schedule = self._extract_schedule(getattr(self.diff_args, "decoding_strategy", "linear"))
        
        modifiable_mask = torch.zeros_like(maskable_mask)
        do_reunmask = getattr(self.diff_args, "reunmask", False)

        # Yield initial state
        xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
        yield xt[0].cpu().numpy(), step_generated, self.diff_args.diffusion_steps, [], []
        
        # Precompute valid digit IDs (1-9)
        valid_digit_ids = [self.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 10)]
        valid_digit_ids = torch.tensor(valid_digit_ids, device=x.device)

        for t in range(self.diff_args.diffusion_steps-1, -1, -1):
            reunmasked_indices = []
            changed_indices = []
            with torch.no_grad():
                if t == self.diff_args.diffusion_steps-1:
                     xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)

                t_tensor = torch.full((batch_size, ), t, device=x.device)
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)

                scores = torch.log_softmax(logits, dim=-1)
                
                # Constrain to digits 1-9 to prevent EOS/Special tokens
                mask_logits = torch.full_like(scores, -float('inf'))
                mask_logits[:, :, valid_digit_ids] = 0
                scores = scores + mask_logits
                
                scores[:, :, self.tokenizer.vocab_size:] = -1000
                probs = scores.exp()
                x0_scores, _ = scores.max(-1)
                sampled_x0 = self._sample_tokens(probs)

                x0 = xt.masked_scatter(maskable_mask, sampled_x0[maskable_mask])
                
                if t > 0:
                    selection_scores = None
                    if denoise_method in ["topk", "downk"]:
                        selection_scores = x0_scores
                    elif denoise_method in ["topk-margin", "downk-margin"]:
                        selection_scores = self._compute_margin_scores(scores)

                    current_mask = maskable_mask.clone()
                    xt, maskable_mask = self._apply_denoise_strategy(
                        xt=xt,
                        x0=x0,
                        selection_scores=selection_scores,
                        maskable_mask=maskable_mask,
                        init_maskable_mask=init_maskable_mask,
                        t=t,
                        schedule=decoding_schedule,
                        method=denoise_method,
                    )
                    
                    if do_reunmask:
                        correction_tokens = logits.argmax(dim=-1)
                        if modifiable_mask.any():
                            reunmasked_indices = modifiable_mask[0].nonzero(as_tuple=True)[0].cpu().numpy()
                            # Check for changes
                            changed_mask = modifiable_mask & (xt != correction_tokens)
                            if changed_mask.any():
                                changed_indices = changed_mask[0].nonzero(as_tuple=True)[0].cpu().numpy()

                        xt = xt.masked_scatter(modifiable_mask, correction_tokens[modifiable_mask])
                        modifiable_mask = current_mask & (~maskable_mask)
                    
                    # Update step_generated for visualization
                    newly_unmasked = current_mask & (~maskable_mask)
                    newly_unmasked_indices = newly_unmasked[0].nonzero(as_tuple=True)[0].cpu().numpy()
                    for idx in newly_unmasked_indices:
                        step_generated[idx] = t

                else:
                    xt = x0
                    if do_reunmask:
                        correction_tokens = logits.argmax(dim=-1)
                        if modifiable_mask.any():
                            reunmasked_indices = modifiable_mask[0].nonzero(as_tuple=True)[0].cpu().numpy()
                            # Check for changes
                            changed_mask = modifiable_mask & (xt != correction_tokens)
                            if changed_mask.any():
                                changed_indices = changed_mask[0].nonzero(as_tuple=True)[0].cpu().numpy()

                        xt = xt.masked_scatter(modifiable_mask, correction_tokens[modifiable_mask])
                    
                    # Update step_generated for final step
                    remaining_mask = maskable_mask
                    remaining_indices = remaining_mask[0].nonzero(as_tuple=True)[0].cpu().numpy()
                    for idx in remaining_indices:
                        step_generated[idx] = 0
            
            yield xt[0].cpu().numpy(), step_generated, t, reunmasked_indices, changed_indices
            time.sleep(sleep_time)

# --- Generation Logic ---

def generate_sudoku_steps(input_str, denoise_method, temperature, diffusion_steps, sleep_time, enable_reunmask, decoding_strategy="linear"):
    load_the_model()
    
    # Prepare input
    input_str = input_str.replace(".", "0")
    if len(input_str) != 81:
        raise ValueError("Input must be 81 characters long")
    
    # Construct Seq2Seq input: Puzzle + SEP + [MASK]*81 + EOS
    puzzle_tokens = [tokenizer.convert_tokens_to_ids(c) for c in input_str]
    sep_token = [tokenizer.sep_token_id]
    solution_mask_tokens = [tokenizer.mask_token_id] * 81
    eos_token = [tokenizer.eos_token_id]
    
    full_input_ids = puzzle_tokens + sep_token + solution_mask_tokens + eos_token
    # src_mask: 1 for puzzle+sep, 0 for solution+eos
    full_src_mask = [1] * (81 + 1) + [0] * (81 + 1)
    
    input_tensor = torch.tensor([full_input_ids]).cuda()
    src_mask_tensor = torch.tensor([full_src_mask]).cuda()
    
    # Initialize step_generated for the full sequence
    step_generated_full = np.full((len(full_input_ids),), -1, dtype=int)
    step_generated_full[:82] = diffusion_steps # Puzzle + SEP are fixed

    global diffusion_args_global
    
    # Override args with UI inputs
    diffusion_args_global.denoise_method = denoise_method
    diffusion_args_global.diffusion_steps = diffusion_steps
    diffusion_args_global.decoding_strategy = decoding_strategy
    diffusion_args_global.reunmask = enable_reunmask
    
    # Create Trainer
    training_args = Seq2SeqTrainingArguments(output_dir="tmp_demo")
    trainer = DemoTrainer(
        diff_args=diffusion_args_global,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        sampling_temperature=temperature
    )
    
    inputs = {
        "input_ids": input_tensor,
        "src_mask": src_mask_tensor
    }
    
    for xt, step_gen, t, reunmasked_indices, changed_indices in trainer.generate_samples_generator(inputs, step_generated_full, sleep_time):
        # Extract the solution part (indices 82 to 162)
        sol_tokens = xt[82:163]
        sol_steps = step_gen[82:163]
        
        # Filter reunmasked_indices to only those in the solution part
        # reunmasked_indices are indices in full_input_ids (0 to 163)
        # We want indices relative to the solution part (0 to 80)
        # So we subtract 82, and keep only those >= 0 and < 81
        sol_reunmasked = []
        for idx in reunmasked_indices:
            if 82 <= idx < 163:
                sol_reunmasked.append(idx - 82)
        
        sol_changed = []
        for idx in changed_indices:
            if 82 <= idx < 163:
                sol_changed.append(idx - 82)
        
        # For visualization, mark clues as fixed (step=diffusion_steps)
        # We check the original input_str
        for i in range(81):
            if input_str[i] != '0':
                sol_steps[i] = diffusion_steps
        
        yield sol_tokens, sol_steps, t, sol_reunmasked, sol_changed

# --- Visualization ---

def render_sudoku(tokens, step_generated, current_t, total_steps, solution_str=None, reunmasked_indices=None, changed_indices=None, input_str=None):
    # tokens: array of token ids
    # step_generated: array of step numbers when generated
    # current_t: current diffusion step
    
    # Validate solution against clues
    if solution_str and len(solution_str) == 81:
        if input_str and len(input_str) == 81:
            for i in range(81):
                if input_str[i] != '0' and input_str[i] != solution_str[i]:
                    solution_str = None
                    break
        else:
            # Fallback: try to validate against fixed tokens if input_str is missing
            # This is less reliable if tokens are masked
            for i in range(81):
                if step_generated[i] == total_steps:
                    val = tokenizer.convert_ids_to_tokens([tokens[i]])[0]
                    if val == tokenizer.mask_token:
                        continue # Skip masked clues
                    if val != solution_str[i]:
                        solution_str = None
                        break

    html = """
    <style>
        .sudoku-container {
            display: inline-block;
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            padding: 12px;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .sudoku-row {
            display: flex;
        }
        .sudoku-cell-new {
            width: 48px;
            height: 48px;
            border: 1px solid #e5e5ea;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 500;
            color: #1d1d1f !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            margin: 2px;
            border-radius: 8px;
        }
        /* Thicker borders for 3x3 blocks - simulated by margin or color */
        .border-right { margin-right: 6px; }
        .border-bottom { margin-bottom: 6px; }
        
    </style>
    <div class='sudoku-container'>
    """
    
    grid_html = ""
    for row in range(9):
        row_html = "<div class='sudoku-row'>"
        for col in range(9):
            idx = row * 9 + col
            token_id = tokens[idx]
            step = step_generated[idx]
            
            # Decode token
            val = tokenizer.convert_ids_to_tokens([token_id])[0]
            if val == tokenizer.mask_token:
                val = ""
            
            # Determine style
            style = ""
            classes = "sudoku-cell-new"
            if (col + 1) % 3 == 0 and col < 8:
                classes += " border-right"
            if (row + 1) % 3 == 0 and row < 8:
                classes += " border-bottom"
                
            # Color logic
            is_error = False
            is_reunmasked = False
            is_changed = False
            
            if reunmasked_indices is not None and idx in reunmasked_indices:
                is_reunmasked = True
            
            if changed_indices is not None and idx in changed_indices:
                is_changed = True

            if solution_str and len(solution_str) == 81:
                correct_char = solution_str[idx]
                if val and val != correct_char:
                    is_error = True

            # Apple-style colors
            # Red: #ff3b30 (Error)
            # Green: #34c759 (Success/Changed)
            # Orange: #ff9500 (Warning/Reunmasked)
            # Gray: #8e8e93 (Fixed)
            # Blue: #0071e3 (Generated)
            
            if is_error:
                style += "background-color: #ffe5e5; color: #ff3b30 !important; font-weight: 700;"
                if is_changed:
                    style += "box-shadow: inset 0 0 0 2px #34c759;"
                elif is_reunmasked:
                    style += "box-shadow: inset 0 0 0 2px #ff9500;"
            elif is_changed:
                style += "background-color: #e8f5e9; color: #1d1d1f !important; box-shadow: inset 0 0 0 2px #34c759;"
            elif is_reunmasked:
                style += "background-color: #fff8e1; color: #1d1d1f !important; box-shadow: inset 0 0 0 2px #ff9500;"
            elif step == total_steps:
                style += "background-color: #f2f2f7; color: #1d1d1f !important; font-weight: 600;"
            elif step == -1:
                # Not generated yet (Mask)
                style += "background-color: white;"
            else:
                # Generated
                # Use opacity of blue to indicate confidence/step
                normalized = step / max(1, total_steps - 1) # 0 to 1
                # 1 (Early) -> Lighter Blue, 0 (Late) -> Darker Blue? 
                # Actually, usually later steps (closer to 0) are more "final".
                # Let's make them all a nice blue, but vary opacity.
                
                # Let's use a solid blue but with varying lightness
                # Blue HSL: 210
                # Lightness: 95% (Early) -> 85% (Late)
                lightness = 96 - (normalized * 12)
                style += f"background-color: hsl(210, 100%, {lightness}%); color: #0071e3 !important;"
                
                if step == current_t:
                    style += "border: 2px solid #0071e3; transform: scale(1.05); z-index: 10;"
            
            row_html += f"<div class='{classes}' style='{style}'>{val}</div>"
        row_html += "</div>"
        grid_html += row_html
        
    html += grid_html + "</div>"
    return html

def render_static_sudoku(sudoku_str, solution_str=None):
    if not sudoku_str:
        return ""
    # Clean string
    sudoku_str = sudoku_str.replace(".", "0")
    if len(sudoku_str) != 81:
        return "<div style='color:red'>Invalid Sudoku String (Length must be 81)</div>"
    
    # Check compatibility of solution
    if solution_str:
        if len(solution_str) != 81:
            solution_str = None
        else:
            for i in range(81):
                if sudoku_str[i] != '0' and sudoku_str[i] != solution_str[i]:
                    solution_str = None
                    break

    html = """
    <style>
        .sudoku-container-static {
            display: inline-block;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 6px;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin-top: 10px;
        }
        .sudoku-row-static {
            display: flex;
        }
        .sudoku-cell-static {
            width: 32px;
            height: 32px;
            border: 1px solid #e5e5ea;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 500;
            background-color: white;
            color: #1d1d1f !important;
            margin: 1px;
            border-radius: 3px;
        }
        .border-right-static { margin-right: 3px; }
        .border-bottom-static { margin-bottom: 3px; }
        .fixed-cell-static { background-color: #f2f2f7; font-weight: 600; }
        .solution-cell-static { color: #8e8e93 !important; }
    </style>
    <div class='sudoku-container-static'>
    """
    
    grid_html = ""
    for row in range(9):
        row_html = "<div class='sudoku-row-static'>"
        for col in range(9):
            idx = row * 9 + col
            char = sudoku_str[idx]
            val = char if char != '0' else ""
            
            classes = "sudoku-cell-static"
            if (col + 1) % 3 == 0 and col < 8:
                classes += " border-right-static"
            if (row + 1) % 3 == 0 and row < 8:
                classes += " border-bottom-static"
            
            if val:
                classes += " fixed-cell-static"
            elif solution_str:
                val = solution_str[idx]
                classes += " solution-cell-static"
            
            row_html += f"<div class='{classes}'>{val}</div>"
        row_html += "</div>"
        grid_html += row_html
        
    html += grid_html + "</div>"
    return html

def load_random_sudoku():
    quiz = "000000000000003085001020000000507000004000100090000000500000073002010000000040009"
    solution = None
    try:
        data_path = os.path.join(os.path.dirname(__file__), "../data/sudoku_test.csv")
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            rows = list(reader)
            if rows:
                random_row = random.choice(rows)
                quiz, solution = random_row[0], random_row[1] # Return quiz and solution
    except Exception as e:
        print(f"Error loading random sudoku: {e}")
    
    return quiz, solution, render_static_sudoku(quiz, solution)

def update_view(slider_val, history, steps, solution_str, input_str=None):
    if not history:
        return None, "No history"
    
    idx = int(slider_val)
    if idx < 0: idx = 0
    if idx >= len(history): idx = len(history) - 1
    
    frame = history[idx]
    
    # Use steps from history if possible to ensure consistent coloring
    total_steps_val = int(steps)
    if history and 't' in history[0]:
         total_steps_val = history[0]['t']

    if input_str:
        input_str = input_str.replace(".", "0")

    html = render_sudoku(
        frame['tokens'], 
        frame['step_generated'], 
        frame['t'], 
        total_steps_val, 
        solution_str, 
        frame['reunmasked_indices'],
        frame.get('changed_indices', []),
        input_str=input_str
    )
    return html, f"Step: {frame['t']}"

def run_demo(input_str, method, temp, steps, sleep_time, solution_str=None):
    enable_reunmask = True
    if not input_str:
        # Default hard sudoku
        input_str = "080050060460907108005000029970006500000872031300049000004025003010000480603100007"
        
    generator = generate_sudoku_steps(input_str, method, temp, int(steps), sleep_time, enable_reunmask)
    
    history = []
    
    input_str_clean = input_str.replace(".", "0")

    for tokens, step_gen, t, reunmasked_indices, changed_indices in generator:
        # Save frame
        frame = {
            'tokens': tokens.copy(),
            'step_generated': step_gen.copy(),
            't': t,
            'reunmasked_indices': reunmasked_indices,
            'changed_indices': changed_indices
        }
        history.append(frame)
        
        html = render_sudoku(tokens, step_gen, t, int(steps), solution_str, reunmasked_indices, changed_indices, input_str=input_str_clean)
        yield html, f"Step: {t}", history, gr.update(maximum=len(history)-1, value=len(history)-1)

# --- Gradio App ---

css = """
.gradio-container {
    max-width: 100% !important; 
    margin: 0 !important; 
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #f5f5f7 !important;
}
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #1d1d1f !important;
    margin-bottom: 10px;
    text-align: center;
}

/* Main Panels */
.main-panel {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.05) !important;
    border-radius: 24px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.04) !important;
    padding: 30px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
    height: 100% !important;
}

/* Equal height row */
.equal-height-row {
    display: flex !important;
    align-items: stretch !important;
    gap: 20px;
}

/* Inputs */
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background-color: #fbfbfd !important;
    border: 1px solid #d2d2d7 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

/* Fix for gray background on controls */
.main-panel .block {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Button */
button.primary {
    background: #0071e3 !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease;
    margin-top: auto !important; /* Push to bottom */
    width: 100%;
}
button.primary:hover {
    background: #0077ed !important;
    transform: scale(1.01);
    box-shadow: 0 4px 12px rgba(0,113,227,0.3) !important;
}

/* Reset internal block styles */
/* Removed to allow specific styling above */

/* Center Sudoku */
.sudoku-container, .sudoku-container-static {
    margin: 0 auto;
    display: table;
}
"""

with gr.Blocks(title="Sudoku Diffusion Generation", css=css) as demo:
    gr.Markdown("# 🧩 Sudoku Diffusion Generation")
    
    # Default sample from csv
    default_quiz = "080050060460907108005000029970006500000872031300049000004025003010000480603100007"
    default_sol = "789251364462937158135468729978316542546872931321549876894725613217693485653184297"

    solution_state = gr.State(default_sol)
    history_state = gr.State([])

    with gr.Row(elem_classes="equal-height-row"):
        # Left Panel
        with gr.Column(scale=1, elem_classes="main-panel"):
            gr.Markdown("### ⚙️ Configuration", elem_id="left-header")
            
            with gr.Row():
                input_sudoku = gr.Textbox(label="Input Sudoku", value=default_quiz, lines=2, scale=3)
                random_btn = gr.Button("🎲 Random", scale=1)
            
            # Preview
            input_html = gr.HTML(label="Preview", value=render_static_sudoku(default_quiz, default_sol))
            
            with gr.Row():
                method = gr.Dropdown(label="Denoise Method", choices=["origin", "topk", "topk-margin", "downk", "downk-margin"], value="topk-margin")
                steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=20, step=1)
            
                with gr.Row():
                    temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.0, step=0.1)
                    sleep_time = gr.Slider(label="Speed (s)", minimum=0.0, maximum=2.0, value=0.1, step=0.1)
                
            btn = gr.Button("Generate Solution", variant="primary")
            
        # Right Panel        # Right Panel
        with gr.Column(scale=1, elem_classes="main-panel"):
            gr.Markdown("### 👁️ Visualization", elem_id="right-header")
            
            status = gr.Label(label="Status", value="Ready")
            output_html = gr.HTML(label="Sudoku Grid")
            step_slider = gr.Slider(label="History Timeline", minimum=0, maximum=0, step=1, value=0)
            
    # Events
    random_btn.click(load_random_sudoku, outputs=[input_sudoku, solution_state, input_html])
    input_sudoku.change(render_static_sudoku, inputs=[input_sudoku, solution_state], outputs=[input_html])
    
    btn.click(run_demo, inputs=[input_sudoku, method, temp, steps, sleep_time, solution_state], outputs=[output_html, status, history_state, step_slider])
    
    step_slider.change(update_view, inputs=[step_slider, history_state, steps, solution_state, input_sudoku], outputs=[output_html, status])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
