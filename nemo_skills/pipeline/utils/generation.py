# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import logging
import os
import shlex
import subprocess
from collections import defaultdict

from nemo_skills.pipeline.utils.cluster import get_tunnel
from nemo_skills.pipeline.utils.mounts import get_unmounted_path
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.utils import get_chunked_filename, get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def get_chunked_rs_filename(
    output_dir: str,
    random_seed: int = None,
    chunk_id: int = None,
) -> str:
    """
    Return a path of the form: {output_dir}/output[-rsSEED][-chunkK].jsonl
    """
    if random_seed is not None:
        base_filename = f"output-rs{random_seed}.jsonl"
    else:
        base_filename = f"output.jsonl"

    # If chunking is enabled, add the chunk suffix
    if chunk_id is not None:
        base_filename = get_chunked_filename(chunk_id, base_filename)
    return os.path.join(output_dir, base_filename)


def get_expected_done_files(output_dir, random_seeds, chunk_ids):
    """
    Returns a mapping of (seed, chunk_id) to expected .done file paths
    """
    file_map = {}
    for seed in random_seeds:
        for chunk_id in chunk_ids:
            output_file = get_chunked_rs_filename(output_dir, random_seed=seed, chunk_id=chunk_id)
            file_map[(seed, chunk_id)] = f"{output_file}.done"
    return file_map


def get_remaining_jobs(cluster_config, output_dir, random_seeds, chunk_ids, rerun_done):
    """
    Determines which jobs still need to be run based on missing .done files.
    Returns a mapping from random_seed to list of chunk_ids that need processing.
    """
    if rerun_done:
        return {seed: copy.deepcopy(chunk_ids) for seed in random_seeds}

    status_dir = get_unmounted_path(cluster_config, output_dir)
    expected_files = get_expected_done_files(output_dir, random_seeds, chunk_ids)
    check_commands = []
    for (seed, chunk_id), filepath in expected_files.items():
        unmounted_path = filepath.replace(output_dir, status_dir)
        # Create identifiers that can be parsed from output
        seed_str = "NONE" if seed is None else str(seed)
        chunk_str = "NONE" if chunk_id is None else str(chunk_id)
        check_commands.append(f'if [ ! -f "{unmounted_path}" ]; then echo "MISSING:{seed_str}:{chunk_str}"; fi')
    # If random_seeds has more than N elements, split commands into groups of N
    request_size = len(check_commands[0]) // 10
    if len(expected_files) > request_size:
        outputs = []
        for i in range(0, len(check_commands), request_size):
            group = check_commands[i : i + request_size]
            command = f"bash -c '{'; '.join(group)}'"
            if cluster_config['executor'] == 'slurm':
                out = get_tunnel(cluster_config).run(command).stdout.strip()
            else:
                out = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("utf-8")
            outputs.append(out)
        output = "\n".join(outputs).strip()
    else:
        command = f"bash -c '{'; '.join(check_commands)}'"
        if cluster_config['executor'] == 'slurm':
            output = get_tunnel(cluster_config).run(command).stdout.strip()
        else:
            output = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("utf-8")

    # Parse results into a mapping of missing jobs
    missing_jobs = defaultdict(list)
    for line in output.splitlines():
        if line.startswith("MISSING:"):
            _, seed_str, chunk_str = line.split(":")
            seed = None if seed_str == "NONE" else int(seed_str)
            chunk = None if chunk_str == "NONE" else int(chunk_str)
            missing_jobs[seed].append(chunk)

    done_jobs = defaultdict(list)
    for seed, chunk_id in expected_files.keys():
        if chunk_id not in missing_jobs[seed]:
            done_jobs[seed].append(chunk_id)

    done_jobs_str = ", ".join(
        [
            (
                f"{seed}"
                if not any(chunk is not None for chunk in chunks)
                else f"{seed} (chunks: {', '.join(str(chunk) for chunk in chunks if chunk is not None)})"
            )
            for seed, chunks in done_jobs.items()
            if chunks
        ]
    )
    missing_jobs_str = ", ".join(
        [
            (
                f"{seed}"
                if not any(chunk is not None for chunk in chunks)
                else f"{seed} (chunks: {', '.join(str(chunk) for chunk in chunks if chunk is not None)})"
            )
            for seed, chunks in missing_jobs.items()
            if chunks
        ]
    )

    if missing_jobs_str:
        # only printing this if there are some missing and some done
        if done_jobs_str:
            LOG.warning(
                "The following jobs are incomplete and will be launched: seeds %s",
                missing_jobs_str,
            )
            LOG.warning(
                "The following jobs are completed and will be skipped (to override set --rerun_done): seeds %s",
                done_jobs_str,
            )
    else:
        LOG.warning("All jobs are completed. No jobs will be launched (to override set --rerun_done).")

    return missing_jobs


def get_generation_cmd(
    output_dir,
    input_file=None,
    input_dir=None,
    extra_arguments="",
    random_seed=None,
    eval_args=None,
    chunk_id=None,
    num_chunks=None,
    preprocess_cmd=None,
    postprocess_cmd=None,
    wandb_parameters=None,
    script: str = 'nemo_skills.inference.generate',
):
    """Construct the generation command for language model inference."""
    if input_file is None and input_dir is None:
        raise ValueError("Either input_file or input_dir must be provided.")
    if input_file is not None and input_dir is not None:
        raise ValueError("Please provide either input_file or input_dir, not both.")

    # in this case we are running on the output of another generate command
    # and doing 1-1 mapping of random seeds
    if input_dir is not None:
        if random_seed is None:
            raise ValueError("If input_dir is provided, random_seed must also be specified.")
        input_file = f"{input_dir}/output-rs{random_seed}.jsonl"

    # First get the unchunked filename for the output file
    output_file = get_chunked_rs_filename(
        output_dir=output_dir,
        random_seed=random_seed,
    )
    cmd = "export HYDRA_FULL_ERROR=1 && "
    cmd += f"python -m {script} ++skip_filled=True ++input_file={input_file} ++output_file={output_file} "
    job_end_cmd = ""

    if random_seed is not None and input_dir is None:  # if input_dir is not None, we default to greedy generations
        cmd += (
            f"    ++inference.random_seed={random_seed} "
            f"    ++inference.temperature=0.7 "
            f"    ++inference.top_k=-1 "
            f"    ++inference.top_p=0.95 "
        )

    if chunk_id is not None:
        cmd += f" ++num_chunks={num_chunks} ++chunk_id={chunk_id} "
        output_file = get_chunked_rs_filename(output_dir, random_seed=random_seed, chunk_id=chunk_id)
        donefiles = []
        # we are always waiting for all chunks in num_chunks, no matter chunk_ids in
        # the current run (as we don't want to merge partial jobs)
        for cur_chunk_id in range(num_chunks):
            filename = get_chunked_rs_filename(output_dir=output_dir, random_seed=random_seed, chunk_id=cur_chunk_id)
            donefile = f"{filename}.done"
            donefiles.append(donefile)

        if job_end_cmd:
            job_end_cmd += f" && touch {donefiles[chunk_id]} "
        else:
            job_end_cmd = f"touch {donefiles[chunk_id]} "

        # getting file name as if there is no chunking since that's where we want to merge
        merged_output_file = get_chunked_rs_filename(output_dir=output_dir, random_seed=random_seed)
        merge_cmd = (
            f"python -m nemo_skills.inference.merge_chunks {merged_output_file} "
            f"{' '.join([f[:-5] for f in donefiles])}"
        )
        if postprocess_cmd:
            postprocess_cmd = shlex.quote(postprocess_cmd)
            merge_cmd = f"{merge_cmd} -- {postprocess_cmd}"
        postprocess_cmd = f"{job_end_cmd} && {merge_cmd}"

    else:  # only writing a single status file
        if job_end_cmd:
            job_end_cmd += f" && touch {output_file}.done "
        else:
            job_end_cmd = f"touch {output_file}.done "

        if postprocess_cmd:
            postprocess_cmd = f"{job_end_cmd} && {postprocess_cmd}"
        else:
            postprocess_cmd = job_end_cmd

    cmd += f" {extra_arguments} "

    if eval_args:
        cmd += (
            f" && python -m nemo_skills.evaluation.evaluate_results "
            f"    ++input_files={output_file} "
            f"    {eval_args} "
        )

    return wrap_cmd(
        cmd=cmd,
        preprocess_cmd=preprocess_cmd,
        postprocess_cmd=postprocess_cmd,
        random_seed=random_seed,
        wandb_parameters=wandb_parameters,
    )


def wrap_cmd(cmd, preprocess_cmd, postprocess_cmd, random_seed=None, wandb_parameters=None):
    if preprocess_cmd:
        if random_seed is not None:
            preprocess_cmd = preprocess_cmd.format(random_seed=random_seed)
        cmd = f" {preprocess_cmd} && {cmd} "
    if postprocess_cmd:
        if random_seed is not None:
               postprocess_cmd = postprocess_cmd.replace("{random_seed}", str(random_seed))
        cmd = f" {cmd} && {postprocess_cmd} "
    if wandb_parameters:
        log_wandb_cmd = (
            f"python -m nemo_skills.inference.log_samples_wandb "
            f"    {wandb_parameters['samples_file']} "
            f"    --name={wandb_parameters['name']} "
            f"    --project={wandb_parameters['project']} "
        )
        if wandb_parameters['group'] is not None:
            log_wandb_cmd += f" --group={wandb_parameters['group']} "
        cmd = f"{cmd} && {log_wandb_cmd} "
    return cmd


def configure_client(
    *,  # Force keyword arguments
    model: str,
    server_type: str,
    server_gpus: int,
    server_nodes: int,
    server_address: str,
    server_args: str,
    server_entrypoint: str | None,
    get_random_port: bool,
    extra_arguments: str,
):
    """
    Utility function to configure a client for the model inference server.

    Args:
        model: Mounted Path to the model to evaluate.
        server_type: String name of the server type.
        server_address: URL of the server hosting the model.
        server_gpus: Number of GPUs to use for the server.
        server_nodes: Number of nodes to use for the server.
        server_args: Additional arguments for the server.
        server_entrypoint: Entry point for the server command (will use default if None).
        get_random_port: Whether to get a random port for the server.
        extra_arguments: Extra arguments to pass to the command.

    Returns:
        A tuple containing:
            - server_config: Configuration for the server.
            - server_address: Address of the server.
            - extra_arguments: Updated extra arguments for the command.
    """
    if server_address is None:  # we need to host the model
        server_port = get_free_port(strategy="random") if get_random_port else 5000
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = f"localhost:{server_port}"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
            "server_entrypoint": server_entrypoint,
            "server_port": server_port,
        }
        extra_arguments = (
            f"{extra_arguments} ++server.server_type={server_type} "
            f"++server.host=localhost ++server.port={server_port} ++server.model={model} "
        )
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments = (
            f"{extra_arguments} ++server.server_type={server_type} "
            f"++server.base_url={server_address} ++server.model={model} "
        )
    return server_config, server_address, extra_arguments
