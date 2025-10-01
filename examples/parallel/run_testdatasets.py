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
    dataset: str = "ioi"
    num_workers: int = 16  # number of test workers
    test_batch_size: int = 16 # number of tests to run concurrently
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
        run_files = task_args.get("run_files", [])
        file_creation_commands = [f"mkdir -p {unique_dir}/graders"]

   

        for filepath in run_files:
            filename= filepath["filename"]
            content= filepath["content"]           
            file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/graders/{filename}
{content}
_EOT_
""")

        file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/graders/{task_args["problem_id"]}.cpp
{task_args["generated_code"]}
_EOT_
""")

        file_creation_commands.append(f"""
cat <<'_EOT_' > {unique_dir}/graders/input.txt
{task_args["test_input"]}
_EOT_
""")

        setup_script = "\n".join(file_creation_commands)
        setup_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(setup_script, language='shell', timeout=120)
        )
        if setup_result.get('stderr'):
            raise Exception(f"File setup failed: {setup_result['stderr']}")

        # 2. Compile the code
        compile_command = f"cd {unique_dir}/graders && chmod +x ./compile && ./compile"
        compile_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(compile_command, language='shell', timeout=120)
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
        run_command = f"cd {unique_dir}/graders && chmod +x ./run && ./run < input.txt && rm -f ../graders/"
        run_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(run_command, language='shell', timeout=120)
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
                worker_sandbox.execute_code(f"rm -rf {unique_dir}", language='shell', timeout=120)
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


def eval_testdatasets(input_files, ref_file, test_file):
    cfg_eval = {}
    cfg_sandbox = {}
    eval_config = IOIEvaluatorConfig(_init_nested=True, **cfg_eval)
    sandbox = LocalSandbox(**cfg_sandbox)
    batch_size = eval_config.test_batch_size
    if not os.path.exists(test_file):
        raise ValueError(f"Failed to find test cases in eval dataset directory: {test_file}")

    with open(test_file) as f:
        metadata = json.load(f)

    if not os.path.exists(ref_file):
        raise ValueError(f"Failed to find test cases in eval dataset directory: {ref_file}")
            

   
    pool = multiprocessing.Pool(processes=batch_size, initializer=init_worker, initargs=(sandbox,))

    for jsonl_file in unroll_files(input_files):
        samples = []
        with open(jsonl_file) as f:
            sample = json.load(f)
            
           
        id = sample['id']   
        ioi_id = sample['ioi_id']
        #this part is bad and should be fixed
        ref_data = None
        with open(ref_file) as f:
            for line in f:
                ref_data_line = json.loads(line)  
                if ref_data_line['ioi_id'] == ioi_id:
                    ref_data = ref_data_line
                    print(f"Found ref data for {id}")
                    break
            if ref_data is None:
                raise ValueError(f"Failed to find ref data for {id} in {ref_file}")
        
          # Output file
        base_json_path, _ = os.path.splitext(jsonl_file)
        output_file = f"{base_json_path}_grader.jsonl"  
        initial_completed = 0
        if os.path.exists(output_file):
            #check if each line is a valid json
            with open(output_file, "rt") as f:
                for line in f:
                    #count the number of lines in the file
                    initial_completed += 1
                    if not json.loads(line):
                        raise ValueError(f"Invalid JSON line in {output_file}: {line}")
            

        print(f"ref_data keys: {list(ref_data.keys())}")
        run_files = ref_data['run_files']
        code_list = sample['code_list']
        print(f"Evaluating {id} {ioi_id}")
        print(f"Run code: {len(code_list)}")
        
        _slice_end = len(code_list)
        _slice_len = max(0, _slice_end)
        for x, code in enumerate(code_list[initial_completed:_slice_end]):
            abs_x = x + initial_completed
            print(f"Evaluating {abs_x}/{_slice_len}")
            completion = add_includes(code, ioi_id)
            add_data = ""
            if ioi_id == "triples":
                if id in {7, 8, 9, 10, 11, 12}:
                    add_data = "1\n"
                else:
                    add_data = "2\n"
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
            total_results = []
            for i in range(0, len(test_items), batch_size):
                batch = test_items[i:i + batch_size]
                tasks = []
                for local_idx, (test_data) in enumerate(batch):
                        task_args = {
                            "generated_code": completion,
                            "problem_id": ioi_id,                         
                            "run_files": run_files,
                            "test_input": add_data + test_data['content'],                            
                        }
                        tasks.append((task_args, local_idx))
                results = pool.starmap(run_test_case, tasks)
                total_results.extend(results)

            final_results = {}
            final_results["run_id"] = abs_x
            final_results["results"] = total_results
            with open(output_file, "at") as f:
                f.write(json.dumps(final_results) + "\n")
                
        
    open(output_file + ".done", "w").close()

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
   
    args = parser.parse_args()

    # Support comma-separated items passed as a single token
    raw_inputs = []
    for token in args.input_files:
        raw_inputs.extend([part for part in token.split(',') if part])

    eval_testdatasets(raw_inputs, args.ref_file, args.test_file)     


if __name__ == "__main__":
    main()
