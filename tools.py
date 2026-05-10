import json
import math
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

# Mock IPython before importing neurogolf_utils (IPython not available outside Kaggle)
if "IPython" not in sys.modules:
    _mock_ipython = types.ModuleType("IPython")
    _mock_display = types.SimpleNamespace(
        display=lambda *a, **k: None,
        FileLink=lambda *a, **k: None,
    )
    _mock_ipython.display = _mock_display
    sys.modules["IPython"] = _mock_ipython
    sys.modules["IPython.display"] = _mock_display

# Mock matplotlib if not available (headless environments)
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

import onnx
import onnxruntime
from crewai.tools import BaseTool

import neurogolf_utils


_SCRATCH_DIR = Path("/tmp/arc_agi_scratch")
_KAGGLE_TASK_DIR = Path("/kaggle/input/competitions/neurogolf-2026")
_LOCAL_TASK_DIR = Path("tasks")


def _load_task(task_num: int) -> dict:
    """Load task JSON, trying Kaggle path first then local fallback."""
    if task_num == 0:
        return neurogolf_utils.load_examples(0)
    for path in (
        _KAGGLE_TASK_DIR / f"task{task_num:03d}.json",
        _LOCAL_TASK_DIR / f"task{task_num:03d}.json",
    ):
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return neurogolf_utils.load_examples(task_num)


class GridStatsTool(BaseTool):
    name: str = "grid_stats"
    description: str = (
        "Extracts structured statistics for an ARC-AGI task. "
        "Call this FIRST — before forming any strategy. "
        "Input: task_num (int) — the integer task number (use 0 for the built-in test task). "
        "Do NOT read any files yourself; this tool loads the task internally. "
        "Returns a JSON string with bounding boxes, translation vectors, symmetry tests, "
        "cross-example color invariants, and per-color histograms."
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_to_array(grid: list) -> np.ndarray:
        """Convert a ragged list-of-lists grid to a padded int32 numpy array."""
        h = len(grid)
        w = max(len(r) for r in grid) if grid else 0
        arr = np.zeros((h, w), dtype=np.int32)
        for i, row in enumerate(grid):
            arr[i, : len(row)] = row
        return arr

    @staticmethod
    def _bounding_boxes(arr: np.ndarray, color: int) -> list:
        """Return list of {label, ymin, ymax, xmin, xmax, area} for each connected component of `color`."""
        mask = arr == color
        labeled, n = ndi.label(mask)
        all_slices = ndi.find_objects(labeled)
        boxes = []
        for lbl in range(1, n + 1):
            slices = all_slices[lbl - 1]
            if slices is None:
                continue
            ys, xs = slices
            area = int(np.sum(labeled[ys, xs] == lbl))
            boxes.append({
                "label": lbl,
                "ymin": int(ys.start),
                "ymax": int(ys.stop - 1),
                "xmin": int(xs.start),
                "xmax": int(xs.stop - 1),
                "area": area,
            })
        return boxes

    @staticmethod
    def _translation_vector(in_arr: np.ndarray, out_arr: np.ndarray, color: int):
        """
        Estimate (dy, dx) translation of `color` mask from input to output via
        2-D normalized cross-correlation peak.  Returns [dy, dx] or null if the
        mask is empty in either grid.
        """
        h = max(in_arr.shape[0], out_arr.shape[0])
        w = max(in_arr.shape[1], out_arr.shape[1])

        def pad(a):
            out = np.zeros((h, w), dtype=np.float32)
            out[: a.shape[0], : a.shape[1]] = a
            return out

        m_in = pad((in_arr == color).astype(np.float32))
        m_out = pad((out_arr == color).astype(np.float32))

        if m_in.sum() == 0 or m_out.sum() == 0:
            return None

        # Cross-correlation via FFT
        F_in = np.fft.fft2(m_in)
        F_out = np.fft.fft2(m_out)
        cross = np.fft.ifft2(F_out * np.conj(F_in)).real
        peak = np.unravel_index(np.argmax(cross), cross.shape)
        dy = int(peak[0]) if peak[0] <= h // 2 else int(peak[0]) - h
        dx = int(peak[1]) if peak[1] <= w // 2 else int(peak[1]) - w
        return [dy, dx]

    @staticmethod
    def _symmetry_tests(in_arr: np.ndarray, out_arr: np.ndarray) -> dict:
        """Test whether output equals a simple geometric transform of input."""
        if in_arr.shape != out_arr.shape:
            return {"flipud": False, "fliplr": False, "rot90_1": False, "rot90_2": False, "rot90_3": False}
        return {
            "flipud":  bool(np.array_equal(np.flipud(in_arr), out_arr)),
            "fliplr":  bool(np.array_equal(np.fliplr(in_arr), out_arr)),
            "rot90_1": bool(np.array_equal(np.rot90(in_arr, 1), out_arr)),
            "rot90_2": bool(np.array_equal(np.rot90(in_arr, 2), out_arr)),
            "rot90_3": bool(np.array_equal(np.rot90(in_arr, 3), out_arr)),
        }

    @staticmethod
    def _histograms(arr: np.ndarray, color: int) -> dict:
        """Row-wise and column-wise sum of the binary mask for `color`."""
        mask = (arr == color).astype(int)
        return {
            "row_sums": mask.sum(axis=1).tolist(),
            "col_sums": mask.sum(axis=0).tolist(),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _run(self, task_num: int) -> str:
        try:
            task = _load_task(int(task_num))
        except Exception as e:
            return f"Task load error for task_num={task_num}: {e}"

        stats: dict = {"examples": []}
        invariant_input_colors: set | None = None
        invariant_output_colors: set | None = None
        shape_ratios: list = []

        # For cross-example static-channel invariance on train examples
        # static_input_masks[color] = True iff that color mask never changed
        static_input_masks: dict[int, np.ndarray | None] = {}   # color -> first seen mask
        static_output_masks: dict[int, np.ndarray | None] = {}
        static_input_unchanged: dict[int, bool] = {}
        static_output_unchanged: dict[int, bool] = {}

        for split in ("train", "test"):
            for ex in task.get(split, []):
                inp_raw = ex.get("input", [])
                out_raw = ex.get("output", [])
                if not inp_raw or not out_raw:
                    continue

                in_arr = self._grid_to_array(inp_raw)
                out_arr = self._grid_to_array(out_raw)
                h_in, w_in = in_arr.shape
                h_out, w_out = out_arr.shape

                in_colors = set(int(v) for v in np.unique(in_arr))
                out_colors = set(int(v) for v in np.unique(out_arr))
                all_colors = sorted(in_colors | out_colors)

                # Basic color mapping (positional coincidence)
                color_map: dict[int, int] = {}
                for r in range(min(h_in, h_out)):
                    for c in range(min(w_in, w_out)):
                        ic = int(in_arr[r, c])
                        oc = int(out_arr[r, c])
                        if ic != oc:
                            color_map[ic] = oc

                # --- Heuristic 1: Bounding boxes per color ---
                bbox_in: dict = {}
                bbox_out: dict = {}
                for col in in_colors:
                    boxes = self._bounding_boxes(in_arr, col)
                    if boxes:
                        bbox_in[str(col)] = boxes
                for col in out_colors:
                    boxes = self._bounding_boxes(out_arr, col)
                    if boxes:
                        bbox_out[str(col)] = boxes

                # --- Heuristic 2: Translation vectors per color ---
                translations: dict = {}
                for col in all_colors:
                    vec = self._translation_vector(in_arr, out_arr, col)
                    if vec is not None:
                        translations[str(col)] = vec

                # --- Heuristic 3: Symmetry / geometric transforms ---
                symmetry = self._symmetry_tests(in_arr, out_arr)

                # --- Heuristic 5: Histograms per active color ---
                histograms_in: dict = {}
                histograms_out: dict = {}
                for col in in_colors:
                    histograms_in[str(col)] = self._histograms(in_arr, col)
                for col in out_colors:
                    histograms_out[str(col)] = self._histograms(out_arr, col)

                # Accumulate cross-example invariance data (train only)
                if split == "train":
                    for col in range(10):
                        m_in = (in_arr == col).astype(np.uint8)
                        m_out = (out_arr == col).astype(np.uint8)

                        if col not in static_input_masks:
                            static_input_masks[col] = m_in
                            static_output_masks[col] = m_out
                            static_input_unchanged[col] = True
                            static_output_unchanged[col] = True
                        else:
                            prev_in = static_input_masks[col]
                            prev_out = static_output_masks[col]
                            # Shape mismatch is only fatal if either mask is non-zero;
                            # two all-zero masks of different sizes are still "color absent everywhere".
                            if prev_in.shape != m_in.shape:
                                if prev_in.any() or m_in.any():
                                    static_input_unchanged[col] = False
                            elif not np.array_equal(prev_in, m_in):
                                static_input_unchanged[col] = False

                            if prev_out.shape != m_out.shape:
                                if prev_out.any() or m_out.any():
                                    static_output_unchanged[col] = False
                            elif not np.array_equal(prev_out, m_out):
                                static_output_unchanged[col] = False

                # Global color tracking
                if invariant_input_colors is None:
                    invariant_input_colors = in_colors
                    invariant_output_colors = out_colors
                else:
                    invariant_input_colors &= in_colors
                    invariant_output_colors &= out_colors

                if h_in > 0 and h_out > 0:
                    shape_ratios.append((h_out / h_in, w_out / w_in))

                stats["examples"].append({
                    "split": split,
                    "input_shape": [h_in, w_in],
                    "output_shape": [h_out, w_out],
                    "input_grid": in_arr.tolist(),
                    "output_grid": out_arr.tolist(),
                    "input_colors": sorted(in_colors),
                    "output_colors": sorted(out_colors),
                    "color_mapping": {str(k): v for k, v in color_map.items()},
                    "new_colors_in_output": sorted(out_colors - in_colors),
                    "removed_colors_in_output": sorted(in_colors - out_colors),
                    "bounding_boxes_input": bbox_in,
                    "bounding_boxes_output": bbox_out,
                    "translation_vectors": translations,
                    "symmetry_transforms": symmetry,
                    "histograms_input": histograms_in,
                    "histograms_output": histograms_out,
                })

        # --- Heuristic 4: Cross-example static channels (train only) ---
        static_input_channels = [
            col for col in range(10)
            if static_input_unchanged.get(col, False) and col in static_input_masks
        ]
        static_output_channels = [
            col for col in range(10)
            if static_output_unchanged.get(col, False) and col in static_output_masks
        ]

        stats["invariant_input_colors"] = sorted(invariant_input_colors or [])
        stats["invariant_output_colors"] = sorted(invariant_output_colors or [])
        stats["shapes_identical"] = all(r == (1.0, 1.0) for r in shape_ratios)
        stats["shape_ratios"] = shape_ratios
        stats["cross_example_static_input_channels"] = static_input_channels
        stats["cross_example_static_output_channels"] = static_output_channels
        stats["onnx_channel_note"] = (
            "ONNX tensor shape is [1,10,30,30]. Channel index == color index (0-9). "
            "One-hot: tensor[0][color][row][col] == 1.0 if that cell has that color."
        )

        return json.dumps(stats, indent=2)


class ExecuteOnnxCodeTool(BaseTool):
    name: str = "execute_onnx_code"
    description: str = (
        "Executes Python code that builds and saves an ONNX model. "
        "Input: python_code (str) — full Python source using onnx.helper that ends with "
        "onnx.save(model, output_path). "
        "Also input: output_path (str) — the path where the model should be saved. "
        "The code must contain a variable named output_path or use the provided path. "
        "Returns: SUCCESS message or EXECUTION_ERROR with stderr."
    )

    def _run(self, python_code: str, output_path: str = "") -> str:
        _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        script_path = _SCRATCH_DIR / "build_model.py"

        # Always strip any output_path definition the LLM may have written, then
        # re-inject the canonical path as the very first line so it wins unconditionally.
        if output_path:
            python_code = re.sub(
                r'^\s*output_path\s*=.*$', '', python_code, flags=re.MULTILINE
            )
            preamble = f'output_path = {repr(output_path)}\n'
        else:
            preamble = ''

        full_code = preamble + python_code
        script_path.write_text(full_code)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "EXECUTION_ERROR: Script timed out after 60 seconds."

        if proc.returncode != 0:
            stderr = proc.stderr[-3000:] if len(proc.stderr) > 3000 else proc.stderr
            return f"EXECUTION_ERROR:\n{stderr}"

        # Verify the output file
        target = output_path or str(_SCRATCH_DIR / "model.onnx")
        if not neurogolf_utils.check_network(target):
            return f"SIZE_ERROR: Model file at {target} failed size/existence check."

        return f"SUCCESS: model saved to {target}"


class AuditNetworkTool(BaseTool):
    name: str = "audit_network"
    description: str = (
        "Validates an ONNX model against ARC-AGI task examples and computes performance metrics. "
        "Input: model_path (str) — path to the .onnx file. "
        "Input: task_num (int) — task number (0 for the built-in test task). "
        "Returns: JSON string with pass/fail status, score, memory, params, and node names."
    )

    def _run(self, model_path: str, task_num: int = 0) -> str:
        result = {
            "passed": False,
            "arc_agi_right": 0,
            "arc_agi_wrong": 0,
            "arc_gen_right": 0,
            "arc_gen_wrong": 0,
            "memory_bytes": None,
            "params": None,
            "score": 1.0,
            "node_names": [],
            "errors": [],
        }

        # Step 1: file existence + size check
        if not neurogolf_utils.check_network(model_path):
            result["errors"].append(f"File check failed for {model_path}")
            return json.dumps(result)

        # Step 2: load and sanitize
        try:
            sanitized = onnx.load(model_path)
        except Exception as e:
            result["errors"].append(f"onnx.load failed: {e}")
            return json.dumps(result)

        for node in sanitized.graph.node:
            if node.output:
                node.name = node.output[0]
            if "kernel_time" in node.name:
                result["errors"].append("Banned string 'kernel_time' found in node name.")
                return json.dumps(result)

        result["node_names"] = [n.name for n in sanitized.graph.node]

        # Step 3: load examples
        try:
            examples = neurogolf_utils.load_examples(task_num)
        except Exception as e:
            result["errors"].append(f"load_examples failed: {e}")
            return json.dumps(result)

        # Step 4: create ONNX Runtime session with profiling
        try:
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            unique_prefix = f"audit_{task_num:03d}_{int(time.time())}"
            options.profile_file_prefix = unique_prefix
            session = onnxruntime.InferenceSession(
                sanitized.SerializeToString(), options
            )
        except onnxruntime.ONNXRuntimeError as e:
            result["errors"].append(f"Session creation failed: {e}")
            return json.dumps(result)

        # Step 5: verify subsets
        arc_agi_examples = examples.get("train", []) + examples.get("test", [])
        arc_gen_examples = examples.get("arc-gen", [])

        r, w, _ = neurogolf_utils.verify_subset(session, arc_agi_examples)
        result["arc_agi_right"] = r
        result["arc_agi_wrong"] = w

        r2, w2, _ = neurogolf_utils.verify_subset(session, arc_gen_examples)
        result["arc_gen_right"] = r2
        result["arc_gen_wrong"] = w2

        # Step 6: score
        try:
            trace_path = session.end_profiling()
            memory, params = neurogolf_utils.score_network(sanitized, trace_path)
            result["memory_bytes"] = memory
            result["params"] = params

            if memory is not None and params is not None and memory >= 0 and params >= 0:
                result["score"] = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
        except Exception as e:
            result["errors"].append(f"score_network failed: {e}")

        total_wrong = result["arc_agi_wrong"] + result["arc_gen_wrong"]
        result["passed"] = total_wrong == 0

        return json.dumps(result, indent=2)
