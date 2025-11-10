import sys

from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

# Get run number from command line argument, default to 0
run_number = int(sys.argv[1]) if len(sys.argv) > 1 else 0

eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=generic/default "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++inference.tokens_to_generate=120000 "
        "++inference.extra_body.reasoning_effort=high "
        "++max_concurrent_requests=1024 "
        "++eval_config.test_file=/workspace/llmcoding/eval_dataset/icpc/icpc24_metadata.json "
    ),
    cluster="iad",
    with_sandbox=True,
    keep_mounts_for_sandbox=True,    expname="icpc_eval_run",
    model="/hf_models/gpt-oss-120b",
    server_type="vllm",
    server_gpus=8,
    num_jobs=100,
    benchmarks="icpc:1000",
    data_dir="/workspace/llmcoding/eval_dataset/",
    output_dir=f"/workspace/results/icpc24/gpt-oss-120b/generation/run_{run_number}/",
    metrics_kwargs=f'{{"cluster_folder": "/workspace/results/icpc24/gpt-oss-120b/generation/run_{run_number}/clusters/"}}',
    split="icpc24",
    time_min="04:00:00",
)
#        "++eval_config.input_file=/workspace/results/icpc25/gpt-oss-120b/gencluster/gpt-oss-120b_100.json "
