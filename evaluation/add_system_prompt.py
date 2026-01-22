from pathlib import Path
from datasets import load_dataset, Value
from copy import deepcopy
import os

# ====== CONFIG =====
BENCHMARKS_DIR = Path("./benchmarks")
OUTPUT_DIR = BENCHMARKS_DIR / "processed"
SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."
# ===================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def add_system_and_instruction(example, dataset_name):
    """Add system prompt and instruction following to the example."""
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    prompt = example.get("prompt", [])
    
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt else []
    
    # Add system prompt if not present
    if not prompt or prompt[0].get("role") != "system" or SYSTEM_PROMPT not in prompt[0].get("content", ""):
        prompt = [sys_msg] + prompt
    
    # Add instruction following to user message if missing
    if len(prompt) > 1 and prompt[1].get("role") == "user":
        user_content = prompt[1].get("content", "")
        if "Let's" not in user_content:
            prompt[1]["content"] = user_content + " " + INSTRUCTION_FOLLOWING
    
    example["prompt"] = prompt
    
    # Set data_source if empty
    if not example.get('data_source') or example['data_source'] == "":
        example['data_source'] = f"test-math-{dataset_name}"
    
    # Zhizhou: cannot convert to str
    # Convert ground_truth to string if needed
    if 'reward_model' in example and 'ground_truth' in example['reward_model']:
        gt = example['reward_model']['ground_truth']
        if not isinstance(gt, str):
            example['reward_model']['ground_truth'] = str(gt)
    
    return example


# Process all parquet files in benchmarks directory
parquet_files = list(BENCHMARKS_DIR.glob("*.parquet"))


parquet_files = [
    "/scratch/10922/zhsha/workspace/rotation-project/POLARIS/evaluation/benchmarks/minerva.parquet",
    "/scratch/10922/zhsha/workspace/rotation-project/POLARIS/evaluation/benchmarks/olympiad.parquet"
]

print(f"Found {len(parquet_files)} parquet files to process")

for parquet_file in parquet_files:

    parquet_file = Path(parquet_file)

    print(f"\nProcessing: {parquet_file.name}")
    
    dataset_name = parquet_file.stem
    output_path = OUTPUT_DIR / f"{dataset_name}_processed.parquet"
    
    # Load dataset
    ds = load_dataset("parquet", data_files=str(parquet_file))
    
    # # Prepare new features with ground_truth as string
    # split_name = list(ds.keys())[0]
    # new_features = deepcopy(ds[split_name].features)
    
    # # Ensure ground_truth is string type
    # if 'reward_model' in new_features and 'ground_truth' in new_features['reward_model']:
    #     new_features["reward_model"]["ground_truth"] = Value("string")
    
    # Process each split
    for split in ds.keys():

        # for data in ds[split]:
        #     gt = data["reward_model"]["ground_truth"]
        #     print(type(gt))
        #     breakpoint()
        #     print()


        # Apply transformations
        ds[split] = ds[split].map(
            lambda x: add_system_and_instruction(x, dataset_name),
            # features=new_features,
            desc=f"Adding system prompt and instruction"
        )
        
        # Save
        ds[split].to_parquet(str(output_path))
        
        # Show preview
        first = ds[split][0]
        print(f"  [{split}] saved to {output_path}")
        print(f"  data_source: {first.get('data_source', 'N/A')}")
        print(f"  ground_truth: {first['reward_model']['ground_truth']} (type: {type(first['reward_model']['ground_truth']).__name__})")
        for i, m in enumerate(first["prompt"][:2]):
            role = m.get("role", "?")
            content = m.get("content", "")
            print(f"  Message {i} ({role}): {content[:100]}{'...' if len(content) > 100 else ''}")

print(f"\nAll files processed and saved to {OUTPUT_DIR}")
