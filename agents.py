import os

import litellm
from crewai import Agent, LLM

from tools import AuditNetworkTool, ExecuteOnnxCodeTool, GridStatsTool


# Tell LiteLLM to retry 429s with Retry-After-aware backoff instead of
# immediately raising, which is what causes the cascading 429 storm.
litellm.num_retries = 6
litellm.retry_policy = litellm.RetryPolicy(RateLimitErrorRetries=6)


_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_FREE_MODEL_FAST = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
_FREE_MODEL_STRONG = "openrouter/qwen/qwen3-coder:free"


def build_llm(role: str) -> LLM:
    if role == "architect":
        model = _FREE_MODEL_STRONG
        temperature = 0.2
    elif role == "analyst":
        model = _FREE_MODEL_FAST
        temperature = 0.3   # systematic pattern-finding, not generative
    else:                   # auditor
        model = _FREE_MODEL_FAST
        temperature = 0.1   # factual scoring, maximum determinism

    return LLM(
        model=model,
        api_key=_OPENROUTER_API_KEY,
        api_base=_OPENROUTER_BASE,
        temperature=temperature,
    )


def create_agents(
    llm_analyst: LLM,
    llm_architect: LLM,
    llm_auditor: LLM,
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
            "\n\n"
            "MANDATORY FIRST ACTION: call grid_stats with the integer task_num before any "
            "reasoning. Never skip this — it provides bounding boxes, translation vectors, "
            "symmetry flags, and cross-example invariants that ground your analysis in facts. "
            "\n\n"
            "ABSOLUTE PROHIBITION ON CELL ITERATION: You NEVER describe a solution in terms "
            "of loops, Python iteration, or cell-by-cell inspection. Phrases like 'for each "
            "row', 'iterate over cells', or 'check each pixel' are forbidden. Every thought "
            "MUST be expressed as a bulk tensor operation on the [1,10,30,30] spatial domain. "
            "Example of CORRECT reasoning: 'Channel 3 of the tensor is shifted 2 rows down "
            "via Pad([0,0,2,0,0,0,0,0]) followed by Slice([0,0,0,0,1,10,28,30], axes=[0,1,2,3]).' "
            "Example of FORBIDDEN reasoning: 'For each row, if color is 3, move it down 2.' "
            "\n\n"
            "You think exclusively in parameter-free ONNX ops: Slice, Gather, Transpose, "
            "Reshape, Where, Equal, Add, Sub, Mul, Pad, Concat, Squeeze, Unsqueeze. "
            "You NEVER suggest Conv, MatMul, or Gemm — they introduce trainable parameters "
            "that collapse the score. "
            "\n\n"
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
            "You write production-quality Python — never pseudocode, never placeholder comments. "
            "\n\n"
            "STRICT RULES you never violate:\n"
            "1. IR version 10, opset 10: use onnx.helper.make_opsetid('', 10)\n"
            "2. Input tensor: name='input', dtype=FLOAT, shape=[1,10,30,30]\n"
            "3. Output tensor: name='output', dtype=FLOAT, shape=[1,10,30,30]\n"
            "4. BANNED ops: LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS — "
            "   any of these in the graph causes IMMEDIATE REJECTION by the scorer.\n"
            "5. STATIC SHAPES ONLY — no dynamic dims, no dim_param, no symbolic shapes. "
            "   A single dynamic dimension is AUTOMATIC DISQUALIFICATION from Neurogolf "
            "   scoring. Every intermediate tensor must have fully-resolved integer dims.\n"
            "6. Name EVERY node with a descriptive string, e.g. name='Slice_Color2_Rows'\n"
            "7. Single graph input ('input'), single graph output ('output')\n"
            "8. No tensor name may collide with any initializer name\n"
            "9. No tensor name may contain 'kernel_time'\n"
            "\n"
            "SCORING: score = max(1.0, 25.0 - ln(memory_bytes + params)). "
            "Favor ops that add ZERO parameters: Slice, Gather, Where, Equal, Transpose, "
            "Reshape, Pad, Concat, Add, Sub, Mul. Avoid Conv/MatMul/Gemm. "
            "\n\n"
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
            "\n\n"
            "You reason over these metrics to write actionable feedback. "
            "You ALWAYS include 'score: X.XX' somewhere in your output (the exact float). "
            "You ALWAYS start your report with 'PASS' or 'FAIL' on the first line. "
            "\n\n"
            "Your feedback structure:\n"
            "- Status: PASS or FAIL\n"
            "- Score: the numeric score from the audit\n"
            "- Accuracy: how many examples passed/failed\n"
            "- If FAIL or score < 15.0: identify specific nodes by name from node_names "
            "  that are causing problems, and suggest STRATEGY-level fixes "
            "  (e.g., 'Node Slice_Color2 outputs wrong shape; the Slice axes parameter "
            "  should be [0,1] not [1,2]'). "
            "\n\n"
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
