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
import json
import argparse
import multiprocessing
import os
import re
import asyncio

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class IOIEvaluatorConfig:
    dataset: str = "ioi24"
    num_workers: int = 4  # number of test workers
    test_batch_size: int = 5  # number of tests to run concurrently
    # where test cases are stored in automatically mounted eval datasets folder.
    test_file: str = "/eval_dataset/ioi24/test_metadata.json"


def init_worker(sandbox_arg):
    global worker_sandbox
    worker_sandbox = sandbox_arg
    # Create and set a dedicated event loop for this worker process.  
    # Re-using the same loop for all subsequent sandbox calls avoids the
    # "Event loop is closed" error that occurs when each call spins up
    # and closes its own loop (as happens with asyncio.run).
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

        file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/graders/{task_args["problem_id"]}.cpp
{task_args["generated_code"]}
_EOT_
""")

        for filepath, content in grader_files:
            dir_name = os.path.dirname(filepath)
            if dir_name:
                file_creation_commands.append(f"mkdir -p {unique_dir}/{dir_name}")
            file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/{filepath}
{content}
_EOT_
""")

        file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/compile.sh
{task_args["compile_code"]}
_EOT_
chmod +x {unique_dir}/compile.sh
""")

        file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/run.sh
{task_args["run_code"]}
_EOT_
chmod +x {unique_dir}/run.sh
""")

        file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/input.txt
{task_args["test_input"]}
_EOT_
""")


        setup_script = "\n".join(file_creation_commands)
        setup_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(setup_script, language='shell')
        )
        if setup_result.get('stderr'):
            raise Exception(f"File setup failed: {setup_result['stderr']}")

        # 2. Compile the code
        compile_command = f"cd {unique_dir} && ./compile.sh"
        compile_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(compile_command, language='shell')
        )

        result = {
            "compile_success": not compile_result.get('stderr'),
            "compile_stdout": compile_result.get('stdout', ''),
            "compile_stderr": compile_result.get('stderr', ''),
            "run_stdout": "",
            "run_stderr": "",
        }

        if not result["compile_success"]:
            return result

        # 3. Run the code
        run_command = f"cd {unique_dir} && ./run.sh"
        run_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(run_command, language='shell')
        )

        run_stdout = run_result.get('stdout', '')
        run_stderr = run_result.get('stderr', '')

        result.update({
            "run_stdout": run_stdout,
            "run_stderr": run_stderr,
        })
        

        return result

    except Exception as e:
        return {"output": "", "error": str(e)}

    finally:
        # 4. Clean up the directory
        # Fire and forget; ignore return values
        try:
            worker_loop.run_until_complete(
                worker_sandbox.execute_code(f"rm -rf {unique_dir}", language='shell')
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
    code_header = '#include <bits/stdc++.h>\n'
    # include the problem header
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + '\n'
    # use namespace std since models forget std:: often
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    return code_header + code


def eval_ioi(input_files, ref_file, test_file):
    cfg_eval = {}
    cfg_sandbox = {}
    eval_config = IOIEvaluatorConfig(_init_nested=True, **cfg_eval)
    sandbox = LocalSandbox(**cfg_sandbox)
    batch_size = eval_config.test_batch_size
    if not os.path.exists(ref_file):
        raise ValueError(f"Failed to find test cases in eval dataset directory: {ref_file}")
    with open(ref_file) as f:
        for line in f:
            ref_data = json.loads(line)            
            

    if not os.path.exists(test_file):
        raise ValueError(f"Failed to find test cases in eval dataset directory: {test_file}")

    with open(test_file) as f:
        metadata = json.load(f)

    pool = multiprocessing.Pool(processes=batch_size, initializer=init_worker, initargs=(sandbox,))

    for jsonl_file in unroll_files(input_files):
        samples = []
        with open(jsonl_file) as f:
            sample = json.load(f)


        
        id = sample['id']
        ioi_id = sample['ioi_id']
        run_code = ref_data['run']
        grader_files = ref_data['grader_files']
        compile_code = ref_data['compile']
        code_list = sample['code_list']
        print(f"Evaluating {id} {ioi_id}")
        print(f"Run code: {len(code_list)}")
        full_results = []
        for x, code in enumerate(code_list[start_idx:min(end_idx, len(code_list))]):
            print(f"Evaluating {x}/{len(code_list[start_idx:min(end_idx, len(code_list))])}")
            completion = add_includes(code, ioi_id)
            # Resolve key in metadata robustly: try numeric id, string id, ioi_id
            metadata_key = None
            if isinstance(metadata, dict):
                for candidate in (id, str(id), ioi_id, str(ioi_id)):
                    if candidate in metadata:
                        metadata_key = candidate
                        break
            if metadata_key is None:
                # Provide a helpful error
                available_keys_preview = list(metadata.keys())[:10] if isinstance(metadata, dict) else type(metadata)
                raise KeyError(
                    f"Unable to find tests for id={id} or ioi_id={ioi_id} in test metadata. "
                    f"Available keys preview: {available_keys_preview}"
                )
            test_items = metadata[metadata_key]
            print(f"Test items: {test_items}")
            for i in range(0, len(test_items), batch_size):
                batch = test_items[i:i + batch_size]
                tasks = []
                for local_idx, (test_data) in enumerate(batch):
                        task_args = {
                            "generated_code": completion,
                            "problem_id": ioi_id,
                            "grader_files": grader_files,
                            "run_code": run_code,
                            "compile_code": compile_code,
                            "test_input": test_data['content'],
                            
                        }
                        tasks.append((task_args, local_idx))
                results = pool.starmap(run_test_case, tasks)
                full_results.extend(results)

        
        with open(f"{jsonl_file}.full_results.jsonl", "wt") as f:
            for result in full_results:
                f.write(json.dumps(result) + "\n")

    pool.close()
    pool.join()


def main():
    parser = argparse.ArgumentParser(description="Evaluate IOI generations against provided test cases.")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+',
        required=True,
        help="Space- or comma-separated list of JSONL files or glob patterns containing generations.",
    )
    parser.add_argument(
        "--ref_file",
        type=str,
        default="/workspace/llmcoding/eval_dataset/ioi24/ref_data.json",
        help="Path to IOI reference JSON (defaults to the dataset's test file).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/workspace/llmcoding/eval_dataset/ioi24/test_metadata.json",
        help="Path to IOI test metadata JSON (defaults to the dataset's test file).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for the evaluation.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=1000,
        help="End index for the evaluation.",
    )
    args = parser.parse_args()

    # Support comma-separated items passed as a single token
    raw_inputs = []
    for token in args.input_files:
        raw_inputs.extend([part for part in token.split(',') if part])

    eval_ioi(raw_inputs, args.ref_file, args.test_file)


if __name__ == "__main__":
    main()
