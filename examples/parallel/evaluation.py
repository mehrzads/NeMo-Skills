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

# If you want to avoid committing your changes, you can use the following override:
# NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python ...

import os
import sys
from pathlib import Path
from typing import List, Dict
import re # Keep re import
import argparse # Import argparse
from nemo_skills.pipeline.cli import wrap_arguments, run_cmd
from nemo_skills.pipeline.eval import eval




codegen_root = "/nemo_run/code/"

# Server settings for merge job (can be minimal if merge.py is lightweight)
server_nodes = 1
server_gpus = 1 # merge.py is likely CPU-bound
merge_time_min = '02:00:00' # Adjust as needed for merge.py runtime

def main( code_input_file: str,  cluster: str):            
    print(f"Code input file provided: {code_input_file}")

    # Use the provided path directly
    code_input_file = Path(code_input_file).absolute()
    code_input_dir = code_input_file.parent
    # Derive a base name from the directory path for job/file naming
    # This assumes the last component of the path is the relevant experiment name
    base_code_expname = code_input_file.name
    print(f"Using base name '{base_code_expname}' derived from path for job/file naming.")

    eval_args = "++eval_type=ioi ++eval_config.dataset=ioi24 ++eval_config.test_file=/workspace/llmcoding/eval_dataset/ioi24/test_metadata.json"
   
    print(f"\n--- Processing Evaluation for Run ---")

    # Define unique names for this filter job and its output file
    eval_job_expname = f"{base_code_expname}_eval_run"
    # Log directory for this specific filter job, inside the main inference output dir
    eval_log_dir = code_input_dir / "eval_logs/"

    print(f"  Evaluation Job Name: {eval_job_expname}")
    print(f"  Evaluation Log Directory: {eval_log_dir}")

    # Command to run filter.py
    eval_command = (
        f"python -m nemo_skills.evaluation.evaluate_results "
        f"    ++input_files={code_input_file} "
        f"    {eval_args}"
    )
    
    

    run_cmd(
        ctx=wrap_arguments(""), # No hydra overrides needed for this simple script dispatch
        cluster=cluster,
        command=eval_command,
        expname=eval_job_expname,
        log_dir=str(eval_log_dir),
        num_nodes=server_nodes,
        num_gpus=server_gpus,
        with_sandbox=True,
        time_min=merge_time_min,
    )
   
    print(f"--- Submitted Evaluation Job for Run --- (Cluster: {cluster})")

    print("\nAll evaluation jobs submitted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation by merging inference outputs.")
    parser.add_argument("--code_input_file", type=str, required=True, 
                        help="The full path to the directory containing inference outputs.")
    parser.add_argument("--cluster", type=str, default="oci-ord-mz", 
                        help="Cluster to run the merge jobs on (e.g., oci-ord-mz, eos-mz). Default: oci-ord-mz")
    
    args, unknown_args = parser.parse_known_args()

    if not args.code_input_file:
        print("Error: Code input file path cannot be empty.")
        sys.exit(1)
        
    main(args.code_input_file, args.cluster)