# SYSTEM ARCHITECTURE & IMPLEMENTATION SPECIFICATION (V2)
**Project:** Autonomous ARC-AGI Solver via ONNX Graph Generation (Neurogolf Challenge)
**Framework:** Python, CrewAI, ONNX

## 1. Project Overview
Your task is to build a multi-agent system using `CrewAI` designed to autonomously solve 100 ARC-AGI puzzles. The system will read JSON puzzle definitions and output highly optimized, statically shaped `.onnx` models. 

This is NOT a standard machine learning training task. It is a **Tensor Optimization and Graph Engineering** problem. The models generated must adhere to extremely strict competition rules provided in a supplementary `neurogolf_utils.py` file.

## 2. Core Technical Constraints (The "Neurogolf" Rules)
The Architect agent must strictly follow these rules when writing `onnx.helper` code. **Any violation results in immediate disqualification.**
* **I/O Format:** Tensors are strictly `FLOAT32` of shape `[1, 10, 30, 30]` (Batch x Channels x Height x Width). Channels represent one-hot encoded colors.
* **Static Shapes Only:** Dynamic sizing is strictly prohibited. 
* **Banned Operators:** `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`, `COMPRESS`.
* **Scoring Function:** `Score = max(1.0, 25.0 - ln(parameters + memory_bytes))`. Maximize score by avoiding trainable weights (like `Conv`) and favoring parameter-free logic (`Gather`, `Slice`, `Where`).

## 3. Multi-Agent CrewAI Architecture & LLM Reasoning Guidelines
You must define three highly specialized CrewAI agents. **CRITICAL:** Do not just create boilerplate agents. Implement them with the specific LLM reasoning and tooling workflows described below.

### Agent 1: The Analyst (Pattern Recognition Specialist)
* **LLM Usage & Workflow:** This agent MUST NOT just blindly read the JSON. You must write a custom pre-processing Tool for this agent that extracts grid statistics (unique colors, static shapes, invariant properties). The Analyst uses its LLM to read the tool's output alongside the JSON, performs "Chain of Thought" reasoning, and then outputs a strategy.
* **Behavior:** It must translate the puzzle logic into a "10-channel spatial masking" strategy (e.g., "Shift channel 2 by +1 on the X-axis and merge via LogicalOr with channel 0").

### Agent 2: The Architect (Graph Engineer)
* **LLM Usage & Workflow:** The Architect's LLM receives the Analyst's strategy. Its sole job is to write executable Python code using `onnx.helper` that builds and saves the `.onnx` model.
* **Behavior:** It must use explicit node naming conventions (e.g., `name='Extract_Red_Channel_Slice'`) so the Auditor can reference them. It only outputs Python code.

### Agent 3: The Auditor (Principal Efficiency Engineer)
* **LLM Usage & Workflow:** The Auditor MUST use a Custom CrewAI Tool that wraps `neurogolf_utils.verify_network` and `score_network`. The LLM receives the raw JSON output from this tool (e.g., validation errors, parameter counts, memory usage). The Auditor's LLM must then *reason* over these metrics to write human-like, strategic feedback.
* **Behavior:** The Auditor acts as a Principal Engineer. **DO NOT let the Auditor dictate exact Python code to the Architect.** It must provide structured, metric-based feedback referencing node names (e.g., "Target node 'Dense_Transformation' uses 1.2M params resulting in a high log penalty. Suggestion: Replace with boolean masking using `Where` and `Equal` ops.").

## 4. Execution Flow (The Loop)
Implement a robust outer and inner loop controller wrapping the CrewAI process:
1. **Outer Loop:** Iterate through the 100 JSON task files.
2. **Inner Loop (Retry Logic):** For each task, allow a maximum of **10 attempts**.
3. **State Management:** Track the `best_score` and `best_model_path` for the current task.
4. **Handoff & Feedback:** If the Auditor fails the model or the score is low, its LLM-generated feedback must be routed back to the Architect for the next attempt.
5. **Exit Condition:** If Score >= Target (e.g., 15.0) and validation passes, break. If 10 attempts are exhausted, accept the `best_model` and move on.

## 5. Required Deliverables
Please write the complete, production-ready Python codebase including:
1. `main.py`: Loop logic, state management, and CrewAI kickoff execution.
2. `agents.py`: Agent definitions with rigorous system prompts enforcing their specific workflows.
3. `tools.py`: Custom CrewAI tools. You MUST wrap `neurogolf_utils.py` functions here for the Auditor, and create a grid-statistics tool for the Analyst. (Assume `neurogolf_utils.py` exists).
4. `tasks.py`: Task definitions explicitly defining the sequential hand-offs and feedback loops between Analyst -> Architect -> Auditor.