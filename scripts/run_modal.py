"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
flashinfer_cache_volume = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)
TRACE_SET_PATH = "/data"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1"})
    .apt_install("ninja-build")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "apache-tvm-ffi")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={
        TRACE_SET_PATH: trace_volume,
        FLASHINFER_CACHE_PATH: flashinfer_cache_volume,
    },
)
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(
            warmup_runs=2,
            iterations=5,
            num_trials=1,
            timeout_seconds=300,
            profile_baseline=True,
        )

    import logging
    import os

    logging.disable(logging.INFO)
    logging.basicConfig(level=logging.WARNING, force=True)

    import torch
    from flashinfer_bench.bench.evaluators import resolve_evaluator
    from flashinfer_bench.bench.utils import make_eval
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.data import EvaluationStatus

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    max_workloads = int(os.environ.get("FIB_MAX_WORKLOADS", "0"))
    if max_workloads > 0:
        workloads = workloads[:max_workloads]

    print(f"Testing 1 solution: {solution.name}", flush=True)
    print(
        f"Config: warmup={config.warmup_runs}, iterations={config.iterations}, "
        f"trials={config.num_trials}, workloads={len(workloads)}",
        flush=True,
    )

    device = "cuda:0"
    torch.cuda.set_device(0)

    registry = BuilderRegistry.get_instance()
    evaluator_cls = resolve_evaluator(definition)
    runnable = registry.build(definition, solution)

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = {definition.name: {}}

    for idx, workload_trace in enumerate(workloads, start=1):
        workload = workload_trace.workload
        print(f"[{idx}/{len(workloads)}] Running workload {workload.uuid[:8]}...", flush=True)
        torch.cuda.empty_cache()

        baseline = evaluator_cls.build_baseline(
            definition=definition,
            workload=workload,
            cfg=config,
            device=device,
            trace_set_root=trace_set.root,
        )
        inputs = [
            [v.clone() if isinstance(v, torch.Tensor) else v for v in inp]
            for inp in baseline.inputs
        ]
        log_path = str(log_dir / f"{solution.name}_{workload.uuid}.log")

        correctness, evaluation = evaluator_cls.check_correctness(
            definition=definition,
            sol_runnable=runnable,
            inputs=inputs,
            ref_outputs=baseline.outputs,
            cfg=config,
            log_path=log_path,
            device=device,
        )
        try:
            flashinfer_cache_volume.commit()
        except Exception:
            pass

        if evaluation is None:
            performance, evaluation = evaluator_cls.eval_performance(
                definition=definition,
                sol_runnable=runnable,
                inputs=inputs,
                ref_mean_latency_ms=baseline.mean_latency_ms,
                cfg=config,
                log_path=log_path,
                device=device,
            )
            if evaluation is None:
                evaluation = make_eval(
                    status=EvaluationStatus.PASSED,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                    performance=performance,
                )

        entry = {
            "status": evaluation.status.value,
            "solution": solution.name,
        }
        if evaluation.performance:
            entry["latency_ms"] = evaluation.performance.latency_ms
            entry["reference_latency_ms"] = evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = evaluation.performance.speedup_factor
        if evaluation.correctness:
            entry["max_abs_error"] = evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = evaluation.correctness.max_relative_error
        results[definition.name][workload.uuid] = entry

        latency = entry.get("latency_ms")
        if latency is None:
            print(f"[{idx}/{len(workloads)}] {workload.uuid[:8]}: {entry['status']}", flush=True)
        else:
            print(
                f"[{idx}/{len(workloads)}] {workload.uuid[:8]}: {entry['status']} | "
                f"{latency:.3f} ms",
                flush=True,
            )

    try:
        flashinfer_cache_volume.commit()
    except Exception:
        pass

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            sol_name = result.get("solution", "")
            print(f"  [{sol_name}] Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor"):
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text(encoding="utf-8"))
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
