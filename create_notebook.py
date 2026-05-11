"""Generates arc_agi_kaggle.ipynb from the crew source files."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def read(name: str) -> str:
    return (HERE / name).read_text()


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

CELL_INSTALL = """\
!pip install -U transformers torch accelerate torchvision torchaudio
!pip install -q crewai==1.14.4 onnx onnxruntime onnx-tool scipy
"""

CELL_LOAD_MODEL = """\
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

_MODEL_ID = "gemma-4-e4b-it"
_GEMMA_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

_processor = AutoProcessor.from_pretrained(_GEMMA_PATH)
_model = AutoModelForCausalLM.from_pretrained(
    _GEMMA_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
_model.eval()
print("Loaded on:", next(_model.parameters()).device)
"""

CELL_GEMMA_LLM = """\
import re as _re_lib
from typing import Any
from pydantic import PrivateAttr
from crewai.llms.base_llm import BaseLLM

_THINK_RE = _re_lib.compile(r"<think>.*?</think>", _re_lib.DOTALL)


class GemmaLocalLLM(BaseLLM):
    \"\"\"crewai BaseLLM subclass wrapping the locally-loaded Gemma 4 E4B model.

    The model and processor globals (_model, _processor) are bound via PrivateAttr
    so they are shared across all three agent instances without reloading.
    crewai uses the ReAct text path because supports_function_calling() is absent.
    \"\"\"

    _gemma_model: Any = PrivateAttr(default=None)
    _gemma_processor: Any = PrivateAttr(default=None)

    def __init__(self, temperature: float = 0.2, **kwargs: Any) -> None:
        super().__init__(model=_MODEL_ID, temperature=temperature, **kwargs)
        # Bind already-loaded globals — no reloading, zero extra VRAM.
        object.__setattr__(self, "_gemma_model", _model)
        object.__setattr__(self, "_gemma_processor", _processor)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        text = self._gemma_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            thinking_budget=512,
        )
        # Wrap text in a list — required by newer transformers processor API.
        inputs = self._gemma_processor(text=[text], return_tensors="pt")
        first_device = next(self._gemma_model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        torch.cuda.empty_cache()
        with torch.no_grad():
            output_ids = self._gemma_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=self.temperature if self.temperature and self.temperature > 0 else 1.0,
                do_sample=bool(self.temperature and self.temperature > 0),
            )

        raw = self._gemma_processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=False,
        )

        # Free GPU memory immediately after decoding.
        del inputs, output_ids
        torch.cuda.empty_cache()

        try:
            parsed = self._gemma_processor.parse_response(raw)
            result = parsed.get("response", raw)
        except Exception:
            result = _THINK_RE.sub("", raw).strip()

        # Honour crewai's "\\nObservation:" stop word (injected per-agent by executor).
        return self._apply_stop_words(result)

    def get_context_window_size(self) -> int:
        return 131072  # Gemma 4 128k context
"""

CELL_SMOKE_TEST = """\
_test_llm = GemmaLocalLLM(temperature=0.1)
_test_llm.stop = ["\\nObservation:"]
_test_out = _test_llm.call([{"role": "user", "content": "Reply with exactly: Final Answer: OK"}])
print(repr(_test_out))
assert "OK" in _test_out or "Final" in _test_out, f"Unexpected output: {_test_out!r}"
print("GemmaLocalLLM smoke test PASSED")
"""


def make_neurogolf_cell() -> str:
    return read("neurogolf_utils.py")


def make_tools_cell() -> str:
    src = read("tools.py")

    # 1. Remove `import types`
    src = re.sub(r'^import types\n', '', src, flags=re.MULTILINE)

    # 2. Remove the IPython mock block (comment + if block)
    src = re.sub(
        r'# Mock IPython before importing neurogolf_utils.*?'
        r'sys\.modules\["IPython\.display"\] = _mock_display\n',
        '',
        src,
        flags=re.DOTALL,
    )

    # 3. Remove the matplotlib mock block
    src = re.sub(
        r'# Mock matplotlib if not available.*?pass\n',
        '',
        src,
        flags=re.DOTALL,
    )

    # 4. Remove `import neurogolf_utils` (already in scope)
    src = re.sub(r'^import neurogolf_utils\n', '', src, flags=re.MULTILINE)

    # 5. Replace `neurogolf_utils.X` with `X` (functions are in notebook global scope)
    src = src.replace('neurogolf_utils.', '')

    # 6. Clean up any resulting double blank lines
    src = re.sub(r'\n{3,}', '\n\n', src)

    return src.strip() + '\n'


def make_agents_cell() -> str:
    return """\
from crewai import Agent


def build_llm(role: str) -> GemmaLocalLLM:
    temps = {"analyst": 0.3, "architect": 0.2, "auditor": 0.1}
    return GemmaLocalLLM(temperature=temps.get(role, 0.2))


def create_agents(
    llm_analyst: GemmaLocalLLM,
    llm_architect: GemmaLocalLLM,
    llm_auditor: GemmaLocalLLM,
) -> dict:
    analyst = Agent(
        role="ARC-AGI Pattern Recognition Specialist",
        goal=(
            "Analyze ARC-AGI puzzle grids and produce a concrete, actionable ONNX "
            "10-channel spatial masking strategy for the Architect to implement."
        ),
        backstory=(
            "You are a world-class expert in visual pattern recognition and tensor algebra. "
            "You deeply understand ARC-AGI grids as [1,10,30,30] FLOAT32 tensors where "
            "channel index equals color index (0-9). Each cell (row, col) of color C has "
            "tensor[0][C][row][col] == 1.0; all other channels at that position are 0. "
            "\\n\\n"
            "MANDATORY FIRST ACTION: call grid_stats with the integer task_num before any "
            "reasoning. Never skip this \\u2014 it provides bounding boxes, translation vectors, "
            "symmetry flags, and cross-example invariants that ground your analysis in facts. "
            "\\n\\n"
            "ABSOLUTE PROHIBITION ON CELL ITERATION: You NEVER describe a solution in terms "
            "of loops, Python iteration, or cell-by-cell inspection. Phrases like 'for each "
            "row', 'iterate over cells', or 'check each pixel' are forbidden. Every thought "
            "MUST be expressed as a bulk tensor operation on the [1,10,30,30] spatial domain. "
            "Example of CORRECT reasoning: 'Channel 3 of the tensor is shifted 2 rows down "
            "via Pad([0,0,2,0,0,0,0,0]) followed by Slice([0,0,0,0,1,10,28,30], axes=[0,1,2,3]).' "
            "Example of FORBIDDEN reasoning: 'For each row, if color is 3, move it down 2.' "
            "\\n\\n"
            "You think exclusively in parameter-free ONNX ops: Slice, Gather, Transpose, "
            "Reshape, Where, Equal, Add, Sub, Mul, Pad, Concat, Squeeze, Unsqueeze. "
            "You NEVER suggest Conv, MatMul, or Gemm \\u2014 they introduce trainable parameters "
            "that collapse the score. "
            "\\n\\n"
            "Your output is a numbered, concrete strategy: "
            "(1) which channels to select, "
            "(2) which spatial transformations to apply (shifts, crops, pads), "
            "(3) how to combine channels (Where, Equal, logical masking), "
            "(4) the final ONNX op sequence in order with intermediate tensor shapes."
        ),
        tools=[GridStatsTool()],
        llm=llm_analyst,
        allow_delegation=False,
        verbose=True,
        max_iter=10,
    )

    architect = Agent(
        role="ONNX Graph Engineer",
        goal=(
            "Translate the Analyst's strategy into executable Python code using onnx.helper "
            "that builds and saves a correct, minimal .onnx model, then run it with "
            "execute_onnx_code to confirm it executes without errors."
        ),
        backstory=(
            "You are a senior ML compiler engineer specializing in ONNX graph construction. "
            "You write production-quality Python \\u2014 never pseudocode, never placeholder comments. "
            "\\n\\n"
            "STRICT RULES you never violate:\\n"
            "1. IR version 10, opset 10: use onnx.helper.make_opsetid('', 10)\\n"
            "2. Input tensor: name='input', dtype=FLOAT, shape=[1,10,30,30]\\n"
            "3. Output tensor: name='output', dtype=FLOAT, shape=[1,10,30,30]\\n"
            "4. BANNED ops: LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS \\u2014 "
            "   any of these in the graph causes IMMEDIATE REJECTION by the scorer.\\n"
            "5. STATIC SHAPES ONLY \\u2014 no dynamic dims, no dim_param, no symbolic shapes. "
            "   A single dynamic dimension is AUTOMATIC DISQUALIFICATION from Neurogolf "
            "   scoring. Every intermediate tensor must have fully-resolved integer dims.\\n"
            "6. Name EVERY node with a descriptive string, e.g. name='Slice_Color2_Rows'\\n"
            "7. Single graph input ('input'), single graph output ('output')\\n"
            "8. No tensor name may collide with any initializer name\\n"
            "9. No tensor name may contain 'kernel_time'\\n"
            "\\n"
            "SCORING: score = max(1.0, 25.0 - ln(memory_bytes + params)). "
            "Favor ops that add ZERO parameters: Slice, Gather, Where, Equal, Transpose, "
            "Reshape, Pad, Concat, Add, Sub, Mul. Avoid Conv/MatMul/Gemm. "
            "\\n\\n"
            "After writing the code, call execute_onnx_code with the full code and "
            "the correct output_path. If execution fails, read the error, fix the code, "
            "and try again. You may attempt fixes up to 4 times."
        ),
        tools=[ExecuteOnnxCodeTool()],
        llm=llm_architect,
        allow_delegation=False,
        verbose=True,
        max_iter=15,
    )

    auditor = Agent(
        role="ONNX Efficiency Principal Engineer",
        goal=(
            "Validate the generated ONNX model using audit_network, then produce "
            "structured, metric-based engineering feedback that guides the next attempt."
        ),
        backstory=(
            "You are a principal engineer who audits ONNX graphs for correctness and efficiency. "
            "You call audit_network and receive a JSON result with: pass/fail status, "
            "right/wrong example counts, memory_bytes, params, score, and node_names. "
            "\\n\\n"
            "You reason over these metrics to write actionable feedback. "
            "You ALWAYS include 'score: X.XX' somewhere in your output (the exact float). "
            "You ALWAYS start your report with 'PASS' or 'FAIL' on the first line. "
            "\\n\\n"
            "Your feedback structure:\\n"
            "- Status: PASS or FAIL\\n"
            "- Score: the numeric score from the audit\\n"
            "- Accuracy: how many examples passed/failed\\n"
            "- If FAIL or score < 15.0: identify specific nodes by name from node_names "
            "  that are causing problems, and suggest STRATEGY-level fixes "
            "  (e.g., 'Node Slice_Color2 outputs wrong shape; the Slice axes parameter "
            "  should be [0,1] not [1,2]'). "
            "\\n\\n"
            "CRITICAL: You do NOT write Python code. You give engineering guidance only. "
            "Reference node names exactly as they appear in the audit JSON node_names list."
        ),
        tools=[AuditNetworkTool()],
        llm=llm_auditor,
        allow_delegation=False,
        verbose=True,
        max_iter=5,
    )

    return {
        "analyst": analyst,
        "architect": architect,
        "auditor": auditor,
    }
"""


def make_tasks_cell() -> str:
    return read("tasks.py")


def make_main_cell() -> str:
    src = read("main.py")

    # 1. Remove IPython mock block
    src = re.sub(
        r'# Mock IPython before any neurogolf_utils import\n'
        r'if "IPython" not in sys\.modules:.*?'
        r'sys\.modules\["IPython\.display"\] = _mock_display\n\n',
        '',
        src,
        flags=re.DOTALL,
    )

    # 2. Remove matplotlib mock block
    src = re.sub(
        r'try:\n    import matplotlib\n    matplotlib\.use\("Agg"\)\nexcept Exception:\n    pass\n\n',
        '',
        src,
    )

    # 3. Remove `import time` and `import types`
    src = re.sub(r'^import time\n', '', src, flags=re.MULTILINE)
    src = re.sub(r'^import types\n', '', src, flags=re.MULTILINE)

    # 4. Remove inter-module imports (already in scope in the notebook)
    src = re.sub(r'^import neurogolf_utils\n', '', src, flags=re.MULTILINE)
    src = re.sub(r'^from agents import build_llm, create_agents\n', '', src, flags=re.MULTILINE)
    src = re.sub(r'^from tasks import create_tasks\n', '', src, flags=re.MULTILINE)
    src = re.sub(r'^from tools import _load_task\n', '', src, flags=re.MULTILINE)

    # 5. Remove rate-limit constants
    src = re.sub(r'^_MAX_RATE_LIMIT_RETRIES = .*\n', '', src, flags=re.MULTILINE)
    src = re.sub(r'^_RATE_LIMIT_BASE_SLEEP = .*\n', '', src, flags=re.MULTILINE)

    # 6. Remove _is_rate_limit() function
    src = re.sub(
        r'\ndef _is_rate_limit\(exc.*?\n(?=\n)',
        '',
        src,
        flags=re.DOTALL,
    )

    # 7. Remove _kickoff_with_backoff() function
    src = re.sub(
        r'\ndef _kickoff_with_backoff\(crew\).*?\n(?=\n)',
        '',
        src,
        flags=re.DOTALL,
    )

    # 8. Replace _kickoff_with_backoff call with direct kickoff
    src = src.replace(
        'result = _kickoff_with_backoff(crew)',
        'result = crew.kickoff()',
    )

    # 9. Set MAX_ATTEMPTS to 3
    src = src.replace('MAX_ATTEMPTS = 1', 'MAX_ATTEMPTS = 3')

    # 10. Remove `if __name__ == "__main__": main()` block (notebook has a dedicated run cell)
    src = re.sub(
        r'\nif __name__ == "__main__":\s*\n    main\(\)\n?',
        '',
        src,
    )

    # 11. Clean up multiple blank lines
    src = re.sub(r'\n{3,}', '\n\n', src)

    return src.strip() + '\n'


CELL_RUN = """\
main()
"""

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

cells = [
    md_cell("# ARC-AGI Crew — Kaggle Notebook (Gemma 4 E4B local)"),
    md_cell("## 1. Install dependencies\n\n`kagglehub` and `transformers` are pre-installed on Kaggle."),
    code_cell(CELL_INSTALL),
    md_cell("## 2. Load Gemma 4 E4B\n\nLoaded once; shared by all three agent LLM instances."),
    code_cell(CELL_LOAD_MODEL),
    md_cell("## 3. GemmaLocalLLM — crewai BaseLLM wrapper"),
    code_cell(CELL_GEMMA_LLM),
    md_cell("## 3b. Smoke test"),
    code_cell(CELL_SMOKE_TEST),
    md_cell("## 4. neurogolf_utils (verbatim)"),
    code_cell(make_neurogolf_cell()),
    md_cell("## 5. Tools"),
    code_cell(make_tools_cell()),
    md_cell("## 6. Agents"),
    code_cell(make_agents_cell()),
    md_cell("## 7. Tasks"),
    code_cell(make_tasks_cell()),
    md_cell("## 8. Main orchestration"),
    code_cell(make_main_cell()),
    md_cell("## 9. Run"),
    code_cell(CELL_RUN),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

out_path = HERE / "arc_agi_kaggle.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Notebook written to: {out_path}")
print(f"Cells: {len(cells)}, Size: {out_path.stat().st_size:,} bytes")
