from pydantic import BaseModel
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
from pathlib import Path

if __name__ == "__main__":
    cluster = "iad"
    model_name = "gpt-oss-120b"
    model_path = f"/hf_models/{model_name}"
    input_file = "/workspace/llmcoding/eval_dataset/icpc25/test.jsonl"
    output_dir = f"/workspace/{model_name}/gencluster/"
    expname = "icpc25_test_case_generations"
    model_path = "/hf_models/gpt-oss-120b"
    server_type = "vllm"
    server_gpus = 8
    server_nodes = 1
    server_args = "--async-scheduling --max-num-seqs=1024"
    num_runs = 1
    generate(
        ctx=wrap_arguments(
            "++skip_filled=True "
            "++prompt_config=/nemo_run/code/recipes/gencluster/prompts/generator.yaml "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++inference.tokens_to_generate=120000 " 
            "++inference.extra_body.reasoning_effort=high "
            "++max_concurrent_requests=1024 "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir+"/generators",
        expname=expname+"_generators",
        model=model_path,
        server_type="vllm",
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        server_args=server_args,
        num_random_seeds=num_runs,
        time_min="04:00:00",
        with_sandbox=True,
    )

    generate(
        ctx=wrap_arguments(
            "++skip_filled=True "
            "++prompt_config=/nemo_run/code/recipes/gencluster/prompts/validator.yaml "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++inference.tokens_to_generate=120000 " 
            "++inference.extra_body.reasoning_effort=high "
            "++max_concurrent_requests=1024 "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir+"/validators",
        expname=expname+"_validators",
        model=model_path,
        server_type="vllm",
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        server_args=server_args,
        num_random_seeds=num_runs,
        time_min="04:00:00",
        with_sandbox=True,
    )

    genrate_tests_command = f"python /nemo_run/code/recipes/gencluster/scripts/extract_cpp_code.py --input_dir {output_dir} ; python /nemo_run/code/recipes/gencluster/scripts/generate_tests.py 1 --min-validators 1 --base-dir {output_dir}"
    run_cmd(
        ctx=wrap_arguments(""), 
        cluster=cluster,
        command=genrate_tests_command,
        expname=expname+"_tests",
        log_dir=str(output_dir+"/test_case_generations_logs"),
        num_nodes=1,
        num_gpus=0,
        with_sandbox=True,
        get_random_port=True,
        run_after=[expname+"_generators", expname+"_validators"],
        exclusive=True,
        time_min="04:00:00",
    )