import dataclasses
import json
import multiprocessing
import os
import re
from dataclasses import field

from nemo_skills.evaluation.evaluator import is_evaluator_registered, register_evaluator
from nemo_skills.utils import nested_dataclass, unroll_files

from nemo_skills.code_execution.sandbox import get_sandbox

import argparse
import sys
from pathlib import Path

@nested_dataclass(kw_only=True)
class IOIEvaluatorConfig:
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=lambda: {'sandbox_type': 'fast'})
    dataset: str = "ioi"
    language: str = 'cpp'  # only cpp is supported currently.
    num_workers: int = 4  # number of test workers
    test_batch_size: int = 5  # number of tests to run concurrently
    # where test cases are stored in automatically mounted eval datasets folder.
    test_file: str = "/workspace/llmcoding/eval_dataset/ioi/examples_metadata.jsonl"


def init_worker(sandbox_arg):
    global worker_sandbox
    worker_sandbox = sandbox_arg

def run_test_case(task_args):
    global worker_sandbox
    try:
        return worker_sandbox.execute_code(**task_args)
    except Exception as e:
        print("worker failed:", e)
        return {
            "score": 0.0,
            "output": "",
            "error": str(e)
        }

def filter_tests(input_file, output_path):
    cfg_eval ={}
    cfg_sandbox = {}
    eval_config = IOIEvaluatorConfig(_init_nested=True, **cfg_eval)
    sandbox = get_sandbox(**eval_config.sandbox)

    assert eval_config.language == 'cpp', "Currently for the IOI benchmark, only C++ is supported."

    batch_size = eval_config.test_batch_size
    if not os.path.exists(eval_config.test_file):
        raise ValueError(f"Failed to find test cases in eval dataset directory: {eval_config.test_file}")

    with open(eval_config.test_file) as f:
        metadata = [json.loads(line) for line in f]

    pool = multiprocessing.Pool(processes=batch_size, initializer=init_worker, initargs=(sandbox,))

    samples = []
    with open(input_file) as f:
        sample = json.load(f)
        samples.append(sample)

        if len(samples) == 0:
            raise ValueError(
                f"No samples found in the file {input_file}.\n"
                f"Make sure the file contains jsonl data with 'codes' key which is a list containing "
                f"individual code samples."
            )

        outputs = []
        for x, entry in enumerate(samples):
            print(f"Evaluating {x}/{len(samples)}")
            print(f"Evaluating jsonl file: {input_file}")
            problem_id = entry['ioi_id']
            
            problem_name = entry.get('name')
            if not problem_name:
                print("Generation is missing a 'name' field.")
                return

            normalized_problem_name = ''.join(e for e in problem_name.lower() if e.isalnum())
            example_data = next((ex for ex in metadata if ''.join(e for e in ex.get('name', '').lower() if e.isalnum()) == normalized_problem_name), None)
            if not example_data:
                print(f"No example found for problem: {problem_name}")
                return

            template_code = example_data.get('template', '')
            if not template_code:
                print(f"Template not found for problem: {problem_name}")
                return
              # Replace a placeholder in the template with the generated code
            if "/* SOLUTION_PLACEHOLDER */" not in template_code:
                print(f"Template {template_path} is missing '/* SOLUTION_PLACEHOLDER */'")
                return
            example_input = example_data.get('input', '')
            if not example_input:
                print(f"Could not parse input from example for {problem_name}")
                return
            example_output = example_data.get('output', '')
            if not example_output:
                print(f"Could not parse output from example for {problem_name}")
                return
            for i, code in enumerate(entry.get('code_list')):
                generated_code = code
                print(f"progress {i}/{len(entry.get('code_list'))}")                          
                full_code = template_code.replace("/* SOLUTION_PLACEHOLDER */", generated_code)
                # Remove any local include statements from the generated code
                full_code = re.sub(r'#include\s*".*?"\s*\n', '', full_code)
                
            

                tasks = []
                task_args = {
                    "generated_code": full_code,
                    "std_input": example_input,
                    "timeout": 10,
                    "language": "cpp"
                }
                tasks.append(task_args)
                results = pool.map(run_test_case, tasks)
                if results[0]["stdout"] == example_output:
                    print("Passed",results[0]["stdout"])            
                else:
                    print("Failed",results[0]["stdout"])            

    pool.close()
    pool.join()

def main():
    parser = argparse.ArgumentParser(description="Merges results from output directories. If 'output_*' subdirectories are found, they are processed individually.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("--output_path", "-o", help="Output file name ", 
                      default="None")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    input_file = Path(args.input_file).absolute()
    if args.output_path == "None":
        output_path = input_file.parent / (input_file.stem + '_filtered_results.jsonl')
    else:
        output_path = Path(args.output_path).absolute()    

    filter_tests(input_file, args.output_path)
            
    
    return 0


if __name__ == "__main__":
    sys.exit(main())