#!/bin/bash

# GPU 설정
export CUDA_VISIBLE_DEVICES=0

# 데이터셋 정보
DATA="electricity"
DATA_PATH="../Dataset/long_term_forecast/electricity/electricity.csv"

# 하이퍼파라미터
BATCH_SIZE=32
LR=1e-5
EPOCHS=100
STRIDE=1
SEQ_LEN=512
LAMB=0.02

# 아키텍처 설정: TILING (unsqueeze -> Linear(1, 512) -> in_chans=C)
ARCH="conv"
SAVE_DIR="./checkpoints/pretrain/pretrain_${DATA}_${ARCH}_lamb"
LOG_DIR="./runs/pretrain_${DATA}_${ARCH}_lamb"

echo "🚀 LeJEPA [${ARCH^^}] Pretraining 시작: ${DATA}"

python run_ts_lejepa.py \
    --arch ${ARCH} \
    --dataset_type ${DATA} \
    --data_path ${DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --stride ${STRIDE} \
    --seq_len ${SEQ_LEN} \
    --lamb ${LAMB} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers 4
