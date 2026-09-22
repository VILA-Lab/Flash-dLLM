# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=512
block_length=16
num_fewshot=0
batch_size=32
threshold=0.9
gamma=0.85
track_num=4
mask_num=4

model_path='GSAI-ML/LLaDA-1.5'

# Create logs directory if it doesn't exist
mkdir -p logs/${task}_batch${batch_size}_length${length}_block${block_length}
log_folder=logs/${task}_batch${batch_size}_length${length}_block${block_length}


# Flash-Cache + Confidence
verify=False
# Get timestamp for unique log naming
timestamp=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist --batch_size ${batch_size} \
--output_path ${log_folder} --log_samples \
--model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},gamma=${gamma},track_num=${track_num},mask_num=${mask_num},show_speed=True,verify=${verify} \
> ${log_folder}/threshold${threshold}_gamma${gamma}_track_num_${track_num}_mask_num=${mask_num}_verify_${verify}_${timestamp}.log

#Flash-Cache + Flash-Verify
verify=True
# Get timestamp for unique log naming
timestamp=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist --batch_size ${batch_size} \
--output_path ${log_folder} --log_samples \
--model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},gamma=${gamma},track_num=${track_num},mask_num=${mask_num},show_speed=True,verify=${verify} \
> ${log_folder}/threshold${threshold}_gamma${gamma}_track_num_${track_num}_mask_num=${mask_num}_verify_${verify}_${timestamp}.log

## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}


