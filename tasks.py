from crewai import Task


def create_tasks(
    agents: dict,
    task_num: int,
    attempt: int,
    model_output_path: str,
    feedback: str,
    best_score: float,
) -> list:
    """
    Create a fresh set of 3 Tasks for one attempt at solving a given ARC-AGI puzzle.

    New Task objects are created each call to avoid CrewAI caching stale descriptions.
    The examples JSON is NOT embedded in templates — tools load it directly by task_num
    to avoid Python str.format_map conflicts with JSON curly braces.
    """
    feedback_section = feedback if feedback else "No previous feedback — this is attempt 1."
    best_score_str = f"{best_score:.3f}"

    analysis_task = Task(
        description=(
            f"You are analyzing ARC-AGI task #{task_num}, attempt {attempt}/10.\n\n"
            f"Step 1: Call the grid_stats tool with task_num={task_num}. "
            f"Do NOT attempt to read files yourself — the tool loads the puzzle internally. "
            f"Your ONLY input to the tool is the integer {task_num}.\n\n"
            f"Step 2: Using the stats output, perform Chain-of-Thought reasoning to identify:\n"
            f"  (a) What spatial transformation maps input grids to output grids?\n"
            f"  (b) Which color channels (0-9) are involved and how?\n"
            f"  (c) Is it a shift, rotation, masking, color replacement, or spatial selection?\n"
            f"  (d) Which ONNX ops implement this with ZERO trainable parameters?\n\n"
            f"CRITICAL: Express ALL reasoning as bulk tensor ops on [1,10,30,30] channel-space. "
            f"NEVER describe a solution using loops, iteration, or cell-by-cell language.\n\n"
            f"Step 3: Produce a numbered strategy listing the exact op sequence with "
            f"intermediate tensor shapes at every step.\n\n"
            f"Previous attempt feedback:\n{feedback_section}"
        ),
        expected_output=(
            "A clear, numbered strategy in natural language describing: "
            "(1) the transformation type, "
            "(2) which channels (0-9) to operate on, "
            "(3) the specific ONNX ops to use (Slice/Gather/Where/Equal/Pad/etc.), "
            "(4) the complete op sequence with intermediate tensor shapes at each step."
        ),
        agent=agents["analyst"],
    )

    architecture_task = Task(
        description=(
            f"You are implementing ARC-AGI task #{task_num}, attempt {attempt}/10 as an ONNX model.\n\n"
            f"Save the model to: {model_output_path}\n\n"
            f"The Analyst has given you a strategy (see context above). Implement it as "
            f"complete, executable Python code using onnx.helper. Requirements:\n"
            f"  - import onnx; from onnx import helper, TensorProto\n"
            f"  - IR version 10: onnx.helper.make_opsetid('', 10)\n"
            f"  - Input: make_tensor_value_info('input', TensorProto.FLOAT, [1,10,30,30])\n"
            f"  - Output: make_tensor_value_info('output', TensorProto.FLOAT, [1,10,30,30])\n"
            f"  - Name every node descriptively, e.g. name='Slice_Extract_Color3'\n"
            f"  - BANNED ops (IMMEDIATE REJECTION): LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, "
            f"FUNCTION, COMPRESS\n"
            f"  - STATIC SHAPES ONLY: every intermediate tensor must have fully-resolved "
            f"integer dims. ANY dynamic dim or dim_param is AUTOMATIC DISQUALIFICATION.\n"
            f"  - End with: onnx.save(model, output_path)  # output_path is pre-set for you\n\n"
            f"After writing the code, call execute_onnx_code with:\n"
            f"  - python_code = <your full code>\n"
            f"  - output_path = {repr(model_output_path)}\n\n"
            f"If execution returns EXECUTION_ERROR, read the error, fix the code, and retry "
            f"(up to 4 times).\n\n"
            f"Auditor feedback from previous attempt:\n{feedback_section}\n"
            f"Best score so far: {best_score_str}"
        ),
        expected_output=(
            f"Confirmation that execute_onnx_code returned SUCCESS and the model is saved "
            f"at {model_output_path}. If all retries failed, report the final error."
        ),
        agent=agents["architect"],
        context=[analysis_task],
    )

    audit_task = Task(
        description=(
            f"You are auditing the ONNX model for ARC-AGI task #{task_num}, attempt {attempt}/10.\n\n"
            f"Call audit_network with:\n"
            f"  - model_path = {repr(model_output_path)}\n"
            f"  - task_num = {task_num}\n\n"
            f"Analyze the JSON result. Your report MUST:\n"
            f"  1. Start with 'PASS' or 'FAIL' on the first line\n"
            f"  2. Include 'score: X.XX' with the exact float score\n"
            f"  3. Report arc_agi_right/wrong and arc_gen_right/wrong counts\n"
            f"  4. If FAIL: identify which examples failed and why (reference node names "
            f"     from the node_names list)\n"
            f"  5. If score < 15.0: name specific nodes driving parameter/memory cost "
            f"     and suggest a strategy-level fix\n\n"
            f"Do NOT write Python code. Write engineering guidance only.\n"
            f"Target score: >= 15.0"
        ),
        expected_output=(
            "A structured audit report containing: "
            "PASS/FAIL on line 1, "
            "'score: X.XX' on line 2, "
            "example accuracy counts, "
            "and (if needed) specific node-level optimization suggestions referencing "
            "exact node names from the audit JSON."
        ),
        agent=agents["auditor"],
        context=[architecture_task],
    )

    return [analysis_task, architecture_task, audit_task]
