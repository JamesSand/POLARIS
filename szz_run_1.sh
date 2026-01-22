



# hanqing setting
model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 4)) t=1.0 p=0.8 k=-1 num_gpus=4 bash szz_eval.sh

model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 8)) t=1.0 p=0.8 k=-1 num_gpus=4 bash szz_eval.sh

model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 16)) t=1.0 p=0.8 k=-1 num_gpus=4 bash szz_eval.sh

# polarise setting
model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 4)) t=1.0 p=1.0 k=20 num_gpus=4 bash szz_eval.sh

model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 8)) t=1.0 p=1.0 k=20 num_gpus=4 bash szz_eval.sh

# just RL setting
model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 4)) t=0.7 p=0.9 k=-1 num_gpus=4 bash szz_eval.sh

model_path="/scratch/10922/zhsha/workspace/rotation-project/POLARIS/fsdp_merged/qwen3-4b-base-adam-1e-6-global_step_200" max_length=$((1024 * 8)) t=0.7 p=0.9 k=-1 num_gpus=4 bash szz_eval.sh




