from pydantic import BaseModel
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
from nemo_skills.inference.parallelgen import ParallelGenTask, ParallelGenConfig



   
class MainConfig(BaseModel):
    model_name: str
    tokens_to_generate: int
    time_min: str
    input_file: str
    output_dir: str
    prompt_config: str    
    inference_temperature: float
    generation_type: str
    num_generations: int
    cluster: str    
def main(config: MainConfig):
   
   
    current_overrides = ( 
        " ++inference.temperature={INFERENCE_TEMPERATURE} "
        " ++inference.tokens_to_generate={TOKENS_TO_GENERATE} "     
        " ++prompt_config={PROMPT_CONFIG} "
        " ++prompt_template={PROMPT_TEMPLATE} "
        " ++num_generations={NUM_GENERATIONS} "
        " ++skip_filled=True "        
    )
    extra_overrides = (
        " ++inference.top_p=1.0 "
        " ++max_concurrent_requests=1024 "
    )

    # Configure server settings based on model_name
    if config.model_name in ["DeepSeek-R1-0528-tp16", "DeepSeek-R1-0528-tp16-sglang"]:
        # SGLang server settings
        prompt_template = "deepseek-instruct"
        server_type = 'sglang'
        server_nodes = 2
        server_gpus = 8
        server_args = "--load-format sharded_state --tensor-parallel-size=16" # Aligned with evaluation.py
    elif config.model_name == "Qwen3-235B-A22B-Thinking-2507-tp16-sglang":
        # SGLang server settings
        prompt_template = "qwen-instruct"
        server_type = 'sglang'
        server_nodes = 2
        server_gpus = 8
        server_args = "--load-format sharded_state --tensor-parallel-size=16" # Aligned with evaluation.py

    elif config.model_name == "gpt-oss-120b":
        # vLLM server settings
        prompt_template = "gpt-oss-high"
        server_type = 'vllm'
        server_nodes = 1
        server_gpus = 8
        server_args = "--async-scheduling --max-num-seqs=1024" # Aligned with evaluation.py
        current_overrides += extra_overrides
    else:
        # vLLM server settings
        prompt_template = "qwen-instruct"
        server_type = 'vllm'
        server_nodes = 1
        server_gpus = 8
        server_args = "--tensor-parallel-size=8" # Aligned with evaluation.py

    
    if config.model_name == "aleks":
        model_path = f"/workspace/hf_models/{config.model_name}" # Access via config object
    else:       
        model_path = f"/hf_models/{config.model_name}" # Access via config object
    print(f"INFO: Using model path: {model_path}")
    
    format_dict = {
        "INFERENCE_TEMPERATURE": config.inference_temperature,
        "TOKENS_TO_GENERATE": config.tokens_to_generate,    
        "PROMPT_CONFIG": config.prompt_config,
        "PROMPT_TEMPLATE": prompt_template,
        "NUM_GENERATIONS": config.num_generations,
    }
     # Check if output_dir is provided in config, otherwise construct it
    if not config.output_dir:
        raise ValueError("output_dir must be provided")


    expname = f"ioi_inference"

    
    expname = expname.format(**format_dict)
    current_overrides = current_overrides.format(**format_dict)
    ctx = wrap_arguments(current_overrides)

 

    if config.model_name == "gpt-oss-120b":
        generate(
            ctx=ctx,
            generation_type=config.generation_type,
            cluster=config.cluster,
            input_file=config.input_file,
            output_dir=config.output_dir,
            expname=expname,
            model=model_path,
            server_type=server_type,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            num_random_seeds=1,
            time_min="01:00:00",
            with_sandbox=True,
        )
    else:
         generate(
            ctx=ctx,
            generation_type=config.generation_type,
            cluster=config.cluster,
            input_file=config.input_file,
            output_dir=config.output_dir,
            expname=expname,
            model=model_path,
            server_type=server_type,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            num_random_seeds=1,
            time_min="01:00:00",
            # set these according to your cluster configuration
            # num_chunks=N,
            # dependent_jobs=M,
        )


if __name__ == "__main__":
    # Define the base overrides common to all modes (Aligned with evaluation.py)
    
    # Get run_num from arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_config", type=str, default="eval/ioi/codegen",
                        help="Prompt configuration to use")
    parser.add_argument("--model_name", type=str, default="Qwen3-8B",
                        help="Model name")
    parser.add_argument("--inference_temperature", type=float, default=0.6,
                        help="Temperature for inference")
    parser.add_argument("--tokens_to_generate", type=int, default=32768,
                        help="Number of tokens to generate")
    parser.add_argument("--input_file", type=str, default="/workspace/llmcoding/eval_dataset/ioi/problems/1.jsonl",
                        help="Input file (used in genselect_generation mode)")
    parser.add_argument("--output_dir", type=str, default="/workspace/test/",
                        help="Output directory")
    parser.add_argument("--generation_type", type=str, default="generate",
                        help="Generation mode")
    parser.add_argument("--num_generations", type=int, default=10,
                        help="Number of generations")
    parser.add_argument("--cluster", type=str, default="oci-ord-mz-1",
                        help="Cluster to use")
   
    args, unknown = parser.parse_known_args()

    main_config_instance = MainConfig(
        model_name=args.model_name,
        tokens_to_generate=args.tokens_to_generate,
        time_min="00:10:00",
        input_file=args.input_file,
        output_dir=args.output_dir,
        prompt_config=args.prompt_config,
        inference_temperature=args.inference_temperature,        
        generation_type=args.generation_type,
        num_generations=args.num_generations,
        cluster=args.cluster,
    )
    main(main_config_instance)