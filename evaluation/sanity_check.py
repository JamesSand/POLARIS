from pathlib import Path
from datasets import load_dataset

# ====== CONFIG =====
PROCESSED_DIR = Path("./benchmarks/processed")
SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
INSTRUCTION_FOLLOWING = "Let's think step by step"
# ===================

def check_dataset(parquet_file):
    """Check a single dataset for compliance."""
    print(f"\n{'='*80}")
    print(f"Checking: {parquet_file.name}")
    print('='*80)
    
    ds = load_dataset("parquet", data_files=str(parquet_file))
    split_name = list(ds.keys())[0]
    dataset = ds[split_name]
    
    total_examples = len(dataset)
    issues = []
    
    for idx, example in enumerate(dataset):
        example_issues = []
        prompt = example.get("prompt", [])
        
        if not isinstance(prompt, list) or len(prompt) == 0:
            example_issues.append(f"  Example {idx}: prompt is not a valid list")
            issues.append('\n'.join(example_issues))
            continue
        
        # Check 1: System prompt exists and is the first message
        system_count = sum(1 for msg in prompt if msg.get("role") == "system")
        
        if system_count == 0:
            example_issues.append(f"  Example {idx}: NO system prompt found")
        elif system_count > 1:
            example_issues.append(f"  Example {idx}: Multiple system prompts found ({system_count})")
        elif prompt[0].get("role") != "system":
            example_issues.append(f"  Example {idx}: System prompt is not the first message")
        elif SYSTEM_PROMPT not in prompt[0].get("content", ""):
            example_issues.append(f"  Example {idx}: System prompt content doesn't match expected")
        
        # Check 2: User message exists and has instruction following
        user_messages = [msg for msg in prompt if msg.get("role") == "user"]
        
        if len(user_messages) == 0:
            example_issues.append(f"  Example {idx}: NO user message found")
        elif len(user_messages) > 1:
            example_issues.append(f"  Example {idx}: Multiple user messages found ({len(user_messages)})")
        else:
            user_content = user_messages[0].get("content", "")
            instruction_count = user_content.count(INSTRUCTION_FOLLOWING)
            
            if instruction_count == 0:
                example_issues.append(f"  Example {idx}: User message missing instruction following")
            elif instruction_count > 1:
                example_issues.append(f"  Example {idx}: Multiple instruction following statements ({instruction_count})")
            elif not user_content.strip().endswith("\\boxed{}."):
                example_issues.append(f"  Example {idx}: User message doesn't end with '\\boxed{{}}'")
        
        if example_issues:
            issues.append('\n'.join(example_issues))
    
    # Report results
    if issues:
        print(f"❌ Found {len(issues)} issues in {total_examples} examples:")
        for issue in issues[:10]:  # Show first 10 issues
            print(issue)
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print(f"✅ All {total_examples} examples passed validation!")
    
    return len(issues) == 0


# Main execution
if not PROCESSED_DIR.exists():
    print(f"Error: Processed directory not found: {PROCESSED_DIR}")
    exit(1)

parquet_files = list(PROCESSED_DIR.glob("*.parquet"))
if not parquet_files:
    print(f"No parquet files found in {PROCESSED_DIR}")
    exit(1)

print(f"Found {len(parquet_files)} processed parquet files")

all_passed = True
for parquet_file in sorted(parquet_files):
    passed = check_dataset(parquet_file)
    all_passed = all_passed and passed

print(f"\n{'='*80}")
if all_passed:
    print("✅ All datasets passed validation!")
else:
    print("❌ Some datasets have issues. Please review the output above.")
print('='*80)




