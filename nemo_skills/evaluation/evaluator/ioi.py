# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import multiprocessing
import os
import re
import sys

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class IOIEvaluatorConfig:
    # Directory where metadata files are located.
    test_dir: str = ""

    # Metadata file name or absolute path (default: {split}_metadata.json).
    test_file: str = "{split}_metadata.json"

    num_workers: int = 4  # number of test workers
    test_batch_size: int = 5  # number of tests to run concurrently


def init_worker(sandbox_arg):
    global worker_sandbox
    worker_sandbox = sandbox_arg
    global worker_loop
    worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(worker_loop)


def run_test_case(task_args: dict, worker_id: int) -> dict:
    global worker_sandbox

    unique_dir = f"/tmp/ioi_run_{worker_id}_{os.getpid()}"

    try:
        # 1. Create all necessary files in one batch command
        grader_files = task_args.get("grader_files", [])
        file_creation_commands = [f"mkdir -p {unique_dir}/graders"]

        file_creation_commands.append(
            f"""
cat <<'_EOT_' > {unique_dir}/graders/{task_args["problem_id"]}.cpp
{task_args["generated_code"]}
_EOT_
"""
        )

        for filepath, content in grader_files:
            dir_name = os.path.dirname(filepath)
            if dir_name:
                file_creation_commands.append(f"mkdir -p {unique_dir}/{dir_name}")
            file_creation_commands.append(
                f"""
cat <<'_EOT_' > {unique_dir}/{filepath}
{content}
_EOT_
"""
            )

        file_creation_commands.append(
            f"""
cat <<'_EOT_' > {unique_dir}/compile.sh
{task_args["compile_code"]}
_EOT_
chmod +x {unique_dir}/compile.sh
"""
        )

        file_creation_commands.append(
            f"""
cat <<'_EOT_' > {unique_dir}/run.sh
{task_args["run_code"]}
_EOT_
chmod +x {unique_dir}/run.sh
"""
        )

        file_creation_commands.append(
            f"""
cat <<'_EOT_' > {unique_dir}/input.txt
{task_args["test_input"]}
_EOT_
"""
        )

        file_creation_commands.append(
            f"""
cat <<'_EOT_' > {unique_dir}/correct_output.txt
{task_args["test_output"]}
_EOT_
"""
        )

        setup_script = "\n".join(file_creation_commands)
        setup_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(setup_script, language="shell", timeout=120)
        )
        if setup_result.get("stderr"):
            raise Exception(f"File setup failed: {setup_result['stderr']}")

        # 2. Compile the code
        compile_command = f"cd {unique_dir} && ./compile.sh"
        compile_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(compile_command, language="shell", timeout=120)
        )

        result = {
            "compile_success": not compile_result.get("stderr"),
            "compile_stdout": compile_result.get("stdout", ""),
            "compile_stderr": compile_result.get("stderr", ""),
            "run_stdout": "",
            "run_stderr": "",
            "score": 0.0,
        }

        if not result["compile_success"]:
            return result

        # 3. Run the code
        run_command = f"cd {unique_dir} && ./run.sh"
        run_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(run_command, language="shell", timeout=120)
        )

        run_stdout = run_result.get("stdout", "")
        run_stderr = run_result.get("stderr", "")

        result.update(
            {
                "run_stdout": run_stdout,
                "run_stderr": run_stderr,
            }
        )

        try:
            result["score"] = float(result["run_stdout"].strip())
        except (ValueError, TypeError):
            result["score"] = 0.0

        return result

    except Exception as e:
        return {"score": 0.0, "output": "", "error": str(e)}

    finally:
        try:
            worker_loop.run_until_complete(
                worker_sandbox.execute_code(f"rm -rf {unique_dir}", language="shell", timeout=120)
            )
        except Exception:
            pass


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


def add_includes(code: str, problem_id: str) -> str:
    """
    Fix common compilation errors for IOI problems.
    """
    if not code:
        return code
    # has most of the useful functions
    code_header = "#include <bits/stdc++.h>\n"
    # include the problem header
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + "\n"
    # use namespace std since models forget std:: often
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    # add missing dummy implementations for IOI 25 triples problem
    dummy = ""
    if problem_id == "triples":
        has_count = re.search(r"\bcount_triples\s*\(", code) is not None
        has_construct = re.search(r"\bconstruct_range\s*\(", code) is not None
        if has_construct and not has_count:
            dummy += "long long count_triples(std::vector<int> H){return 0LL;}\n"
        elif has_count and not has_construct:
            dummy += "std::vector<int> construct_range(int M,int K){return {};}\n"
    return code_header + code + ("\n" + dummy if dummy else "")


def _print_progress_bar(completed_count: int, total_count: int) -> None:
    try:
        completed_count = max(0, min(completed_count, total_count))
        bar = "x" * completed_count + "." * (total_count - completed_count)
        sys.stdout.write(f"[{bar}] {completed_count}/{total_count}\r")
        sys.stdout.flush()
    except Exception:
        # Best-effort progress bar; ignore any stdout errors
        pass


def eval_ioi(cfg):
    eval_config = IOIEvaluatorConfig(_init_nested=True, **cfg.eval_config)
    sandbox = LocalSandbox()
    batch_size = eval_config.test_batch_size

    # Resolve metadata path.
    if not (os.path.isabs(eval_config.test_file) and os.path.exists(eval_config.test_file)):
        fname = eval_config.test_file.format(split=cfg.split)
        search_dir = None
        if eval_config.test_dir:
            search_dir = eval_config.test_dir
        if cfg.data_dir:
            search_dir = os.path.join(cfg.data_dir, "ioi24")
        if search_dir is None:
            raise ValueError("Either data_dir or eval_config.test_dir must be specified.")
        eval_config.test_file = os.path.join(search_dir, fname)

    if not os.path.exists(eval_config.test_file):
        raise ValueError(
            f"Could not find tests file at {eval_config.test_file}. "
            "Provide an absolute path as eval_config.test_file, eval_config.test_dir, or data_dir."
        )

    with open(eval_config.test_file) as f:
        metadata = json.load(f)

    pool = multiprocessing.Pool(processes=batch_size, initializer=init_worker, initargs=(sandbox,))

    for jsonl_file in unroll_files(cfg.input_files):
        samples = []
        with open(jsonl_file) as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)

        if len(samples) == 0:
            raise ValueError(
                f"No samples found in the file {jsonl_file}.\n"
                f"Make sure the file contains jsonl data with 'codes' key which is a list containing "
                f"individual code samples."
            )

        # Progress/status handling
        status_file = f"{jsonl_file}.eval"
        initial_completed = 0
        if os.path.exists(status_file):
            try:
                with open(status_file, "rt") as sf:
                    content = sf.read().strip()
                    # Expect formats like "12/250"; fall back to first integer found
                    m = re.search(r"(\d+)\s*/\s*(\d+)", content)
                    if m:
                        initial_completed = int(m.group(1))
                    else:
                        m2 = re.search(r"\d+", content)
                        if m2:
                            initial_completed = int(m2.group(0))
                        else:
                            initial_completed = 0
            except Exception:
                initial_completed = 0
        initial_completed = max(0, min(initial_completed, len(samples)))

        # Initial progress bar draw
        _print_progress_bar(initial_completed, len(samples))

        outputs = []
        # Track which indices we updated so we can write back correctly on resume
        updated_indices_to_results = {}
        processed_since_start = 0

        for x, entry in enumerate(samples[initial_completed:], start=initial_completed):
            print(f"Evaluating {x}/{len(samples)}")
            completion = extract_final_cpp_block(entry["generation"])
            completion = add_includes(completion, entry["ioi_id"])

            test_case_results = {}
            problem_name = entry["name"]
            problem_metadata = metadata[problem_name]
            for subtask, subtask_data in problem_metadata.items():
                tests = subtask_data["tests"]
                subtask_score = subtask_data["score"]
                subtask_score_precision = subtask_data["score_precision"]
                subtask_passed = True
                subtask_outputs = []
                test_items = list(tests.items())

                scores = []
                for i in range(0, len(test_items), batch_size):
                    batch = test_items[i : i + batch_size]
                    tasks = []
                    for local_idx, (test_name, test_data) in enumerate(batch):
                        task_args = {
                            "generated_code": completion,
                            "problem_id": entry["ioi_id"],
                            "grader_files": entry["grader_files"],
                            "run_code": entry["run"],
                            "compile_code": entry["compile"],
                            "test_input": test_data["input"],
                            "test_output": test_data["output"],
                        }
                        tasks.append((task_args, local_idx))
                    results = pool.starmap(run_test_case, tasks)

                    for (test_name, _), result in zip(batch, results):
                        result_with_name = dict(result)
                        result_with_name["test_name"] = test_name
                        subtask_outputs.append(result_with_name)
                        scores.append(float(result["score"]))
                        if float(result["score"]) == 0.0:
                            # break early as we failed this test case.
                            subtask_passed = False
                            break
                    if not subtask_passed:
                        break

                effective_score = round(min([score for score in scores]) * subtask_score, subtask_score_precision)
                test_case_results[subtask] = {"score": effective_score, "outputs": subtask_outputs}

            outputs.append(
                {
                    "name": entry["name"],
                    "subtask": entry["subtask"],
                    "test_case_results": test_case_results,
                }
            )
            updated_indices_to_results[x] = test_case_results

            # Update status file and progress bar
            try:
                with open(status_file, "wt") as sf:
                    sf.write(f"{initial_completed + processed_since_start}/{len(samples)}")
            except Exception:
                pass
            processed_since_start += 1
            _print_progress_bar(initial_completed + processed_since_start, len(samples))
        
        # Write done file
        open(status_file + ".done", "w").close()
        # delete status file
        os.remove(status_file)
        # Finish progress bar line with newline
        if len(samples) > 0:
            sys.stdout.write("\n")

        # Write only updated indices back to samples (supports resume)
        for idx, result in updated_indices_to_results.items():
            samples[idx]["test_case_results"] = result
        jdump(samples, jsonl_file, mode="wt")

        total_passed = 0
        total_problems = len(outputs)
        for o in outputs:
            for subtask_result in o["test_case_results"].values():
                if subtask_result["score"] > 0:
                    total_passed += 1
        if total_problems > 0:
            try:
                example_name = outputs[0]["name"]
                total_subtasks_per_problem = len(metadata[example_name])
            except Exception:
                total_subtasks_per_problem = 0
            print(f"Subtasks passed: {total_passed} out of {total_problems * total_subtasks_per_problem}")
        else:
            print("Subtasks passed: 0 out of 0")

    pool.close()
    pool.join()
