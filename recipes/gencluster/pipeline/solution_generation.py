from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=generic/default "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++inference.tokens_to_generate=120000 "
        "++inference.extra_body.reasoning_effort=high "
        "++max_concurrent_requests=1024 "
        "++eval_config.test_file=/workspace/llmcoding/eval_dataset/icpc25/test_metadata.json "
        "++eval_config.input_file=/workspace/gpt-oss-120b/gencluster/gpt-oss-120b_100.json "
    ),
    cluster="iad",
    with_sandbox=True,
    keep_mounts_for_sandbox=True,
    expname="icpc_eval_run",
    model="/hf_models/gpt-oss-120b",
    server_type="vllm",
    server_gpus=8,
    num_jobs=100,
    benchmarks="icpc25:1000",
    data_dir="/workspace/llmcoding/eval_dataset/",
    output_dir="/workspace/gpt-oss-120b/generation/run_1_icpc25_1000/",
    extra_metrics_arguments='{"cluster_folder": "/workspace/gpt-oss-120b/generation/run_1_icpc25_1000/clusters/"}',
    split="test",
    time_min="04:00:00",
)
