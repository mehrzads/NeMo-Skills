# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import asyncio
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class CCCEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.json"
    num_workers: int = 16
    test_batch_size: int = 16
    time_scale: float = 1.0
    overwrite: bool = False


_precompile_loop_tls = threading.local()
_test_loop_tls = threading.local()
worker_sandbox = None  # type: ignore


def _sandbox_exec_sync(sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    loop = getattr(_precompile_loop_tls, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _precompile_loop_tls.loop = loop
    return loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def _test_exec_sync(sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    loop = getattr(_test_loop_tls, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _test_loop_tls.loop = loop
    return loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def wait_for_sandbox(sandbox, timeout: int = 240, poll: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = _sandbox_exec_sync(sandbox, "echo hello world", language="shell", timeout=10)
            if resp.get("stdout", "").strip() == "hello world":
                return
        except Exception:
            pass
        time.sleep(poll)
    raise RuntimeError(f"Sandbox not ready after waiting {timeout}s")


def _precompile_problem(problem_id: str, grader_files, compile_code: str, run_code: str, sandbox: LocalSandbox) -> str:
    if getattr(sandbox, "_owner_tid", None) != threading.get_ident():
        sandbox = LocalSandbox()
        wait_for_sandbox(sandbox)
        sandbox._owner_tid = threading.get_ident()

    pre_dir = f"/nemo_run/ccc_pre_{problem_id}_{os.getpid()}"
    os.makedirs(os.path.join(pre_dir, "graders"), exist_ok=True)

    for filepath, content in grader_files:
        target_path = os.path.join(pre_dir, filepath)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

    for script_name, script_content in (("compile.sh", compile_code), ("run.sh", run_code)):
        script_path = os.path.join(pre_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

    _sandbox_exec_sync(sandbox, f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)
    return pre_dir


def run_test_case(task_args: dict, worker_id: int) -> dict:
    unique_dir = f"/nemo_run/ccc_run_{worker_id}_{os.getpid()}_{time.time_ns()}"
    try:
        precompiled_dir = task_args.get("precompiled_dir")
        os.makedirs(unique_dir, exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "graders"), exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "tmp"), exist_ok=True)
        if precompiled_dir and os.path.isdir(precompiled_dir):
            shutil.copytree(precompiled_dir, unique_dir, dirs_exist_ok=True)
        with open(os.path.join(unique_dir, "graders", f"{task_args['problem_id']}.cpp"), "w", encoding="utf-8") as f:
            f.write(task_args["generated_code"])
        with open(os.path.join(unique_dir, "input.txt"), "w", encoding="utf-8") as f:
            f.write(task_args["test_input"])
        with open(os.path.join(unique_dir, "correct_output.txt"), "w", encoding="utf-8") as f:
            f.write(task_args["test_output"])

        sandbox = LocalSandbox()
        compile_result = _test_exec_sync(sandbox, f"cd {unique_dir} && ./compile.sh", language="shell", timeout=120)
        result = {
            "compile_success": not compile_result.get("stderr"),
            "compile_stdout": compile_result.get("stdout", ""),
            "compile_stderr": compile_result.get("stderr", ""),
            "run_stdout": "",
            "run_stderr": "",
            "error": "",
            "score": 0.0,
        }
        if not result["compile_success"]:
            return result

        run_timeout = max(1, int(120 * float(task_args.get("time_scale", 1.0))))
        run_result = _test_exec_sync(
            sandbox,
            f"cd {unique_dir} && export TMPDIR={unique_dir}/tmp && TIME_LIMIT_SCALE={task_args.get('time_scale', 1.0)} ./run.sh",
            language="shell",
            timeout=run_timeout,
        )
        result["run_stdout"] = run_result.get("stdout", "")
        result["run_stderr"] = run_result.get("stderr", "")
        try:
            result["score"] = float(result["run_stdout"].strip())
        except (ValueError, TypeError):
            result["score"] = 0.0
        return result
    except Exception as e:
        return {"score": 0.0, "output": "", "error": str(e)}
    finally:
        try:
            shutil.rmtree(unique_dir, ignore_errors=True)
        except Exception:
            pass


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


def add_includes(code: str, problem_header_include: str | None = None, problem_id: str | None = None) -> str:
    if not code:
        return code
    code_header = "#include <bits/stdc++.h>\n"
    if problem_header_include:
        header_include = f'#include "{problem_header_include}"'
        if header_include not in code:
            code_header += header_include + "\n"
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    dummy = ""
    if problem_id == "triples":
        has_count = re.search(r"\bcount_triples\s*\(", code) is not None
        has_construct = re.search(r"\bconstruct_range\s*\(", code) is not None
        if has_construct and not has_count:
            dummy += "long long count_triples(std::vector<int> H){return 0LL;}\n"
        elif has_count and not has_construct:
            dummy += "std::vector<int> construct_range(int M,int K){return {};}\n"
    return code_header + code + (("\n" + dummy) if dummy else ("\n" if not code.endswith("\n") else ""))


class CCCEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        self.eval_cfg = CCCEvaluatorConfig(_init_nested=True, **config)
        self.sandbox = None
        self.metadata = None
        self.precompiled_cache = {}
        self.test_semaphore = None

    async def _initialize_runtime(self):
        if self.sandbox is not None:
            return

        def _setup():
            sbox = LocalSandbox()
            wait_for_sandbox(sbox)
            sbox._owner_tid = threading.get_ident()
            if not os.path.exists(self.eval_cfg.test_file):
                raise FileNotFoundError(f"Metadata file {self.eval_cfg.test_file} does not exist.")
            with open(self.eval_cfg.test_file, "r", encoding="utf-8") as f:
                metadata_local = json.load(f)
            return sbox, metadata_local

        self.sandbox, self.metadata = await asyncio.to_thread(_setup)
        self.test_semaphore = asyncio.Semaphore(max(1, int(self.eval_cfg.test_batch_size)))

    def _get_precompiled_dir(self, problem_id: str, problem_metadata: dict):
        if problem_id in self.precompiled_cache:
            cached = self.precompiled_cache[problem_id]
            return cached["grader"] if isinstance(cached, dict) else cached

        grader_dir = _precompile_problem(
            problem_id,
            problem_metadata["grader_files"],
            problem_metadata["compile"],
            problem_metadata["run"],
            self.sandbox,
        )
        self.precompiled_cache[problem_id] = {"grader": grader_dir}
        return grader_dir

    def _build_test_task(self, problem_id: str, pre_dir: str, completion: str, test_data: dict):
        return {
            "generated_code": completion,
            "problem_id": problem_id,
            "precompiled_dir": pre_dir,
            "test_input": test_data["input"],
            "test_output": test_data["output"],
            "time_scale": self.eval_cfg.time_scale,
        }

    async def _run_test_task(self, task: dict, worker_id: int) -> dict:
        if self.test_semaphore is None:
            raise RuntimeError("CCC evaluator runtime is not initialized.")
        async with self.test_semaphore:
            return await asyncio.to_thread(run_test_case, task, worker_id)

    def _aggregate_subtask_score(self, subtask_meta: dict, outputs: list[dict]) -> float:
        aggregation = subtask_meta["aggregation"]
        if aggregation == "min":
            scores = [float(out.get("score", 0.0)) for out in outputs]
            return round(
                (min(scores) if scores else 0.0) * float(subtask_meta["score"]),
                int(subtask_meta.get("score_precision", 0)),
            )
        if aggregation == "sum_tests":
            return float(sum(1 for out in outputs if float(out.get("score", 0.0)) > 0.0))
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    async def _evaluate_entry(self, entry: dict) -> dict:
        await self._initialize_runtime()

        problem_id = entry["problem_id"]
        problem_metadata = self.metadata[problem_id]
        completion = add_includes(
            extract_final_cpp_block(entry["generation"]),
            problem_metadata.get("problem_header_include"),
            problem_id,
        )
        pre_dir = await asyncio.to_thread(self._get_precompiled_dir, problem_id, problem_metadata)

        all_test_items = list(problem_metadata["all_tests"].items())
        tasks = []
        for idx, (test_name, test_data) in enumerate(all_test_items):
            task = self._build_test_task(problem_id, pre_dir, completion, test_data)
            tasks.append(self._run_test_task(task, idx))
        results = await asyncio.gather(*tasks)
        test_outputs = {}
        for (test_name, _), result in zip(all_test_items, results):
            result["test_name"] = test_name
            test_outputs[test_name] = result

        test_case_results = {}
        for subtask_name, subtask_meta in problem_metadata["subtasks"].items():
            outputs = [dict(test_outputs[test_name]) for test_name in subtask_meta["test_names"]]
            test_case_results[subtask_name] = {
                "score": self._aggregate_subtask_score(subtask_meta, outputs),
                "outputs": outputs,
            }


        return {
            "name": entry["name"],
            "subtask": entry["subtask"],
            "test_case_results": test_case_results,
        }

    async def eval_full(self, input_files):  # type: ignore[override]
        for jsonl_file in unroll_files(input_files):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                all_samples = [json.loads(line) for line in f]
            outputs = await asyncio.gather(*[self._evaluate_entry(s) for s in all_samples])
            for s, o in zip(all_samples, outputs):
                s["test_case_results"] = o["test_case_results"]
            jdump(all_samples, jsonl_file, mode="wt")

    async def eval_single(self, data_point: dict):
        return await self._evaluate_entry(data_point)
