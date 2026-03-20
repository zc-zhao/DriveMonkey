set -x

NNODES=$1
GPUS=$2
OUTPUT_DIR=$3

# GPUS=${GPUS:-8}
PORT=${PORT:-34229}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# OUTPUT_DIR='work_dirs/internvl_chat_v2_0/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "./weight/InternVL2-4B" \
  --conv_style "phi3-chat" \
  --output_dir ${OUTPUT_DIR} \
  --use_nuscene True \
  --meta_path "./shell/internvl_nuscene_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --pad2square False \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
