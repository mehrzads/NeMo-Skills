import sys

from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

# Get run number and ICPC year from command line arguments
run_number = int(sys.argv[1]) if len(sys.argv) > 1 else 0
icpc_year = int(sys.argv[2]) if len(sys.argv) > 2 else 25

eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=generic/default "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++inference.tokens_to_generate=120000 "
        "++inference.extra_body.reasoning_effort=high "
        "++max_concurrent_requests=1024 "
        f"++eval_config.test_file=/workspace/llmcoding/eval_dataset/icpc/icpc{icpc_year}_metadata.json "
        f"++eval_config.input_file=/workspace/results/icpc{icpc_year}/gpt-oss-120b/gencluster/gpt-oss-120b_100.json "
    ),
    cluster="iad",
    with_sandbox=True,
    keep_mounts_for_sandbox=True,    
    expname="icpc_eval_run",
    model="/hf_models/gpt-oss-120b",
    server_type="vllm",
    server_gpus=8,
    num_jobs=100,
    benchmarks="icpc:1000",
    data_dir="/workspace/llmcoding/eval_dataset/",
    output_dir=f"/workspace/results/icpc{icpc_year}/gpt-oss-120b/generation/run_{run_number}/",
    metrics_kwargs=f'{{"cluster_folder": "/workspace/results/icpc{icpc_year}/gpt-oss-120b/generation/run_{run_number}/clusters/"}}',
    split=f"icpc{icpc_year}",
    time_min="04:00:00",
)
