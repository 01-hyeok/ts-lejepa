#!/bin/bash

# GPU 설정
export CUDA_VISIBLE_DEVICES=1

# 데이터셋 정보
DATA="tsld"
DATA_PATH="../Dataset/TSLD-1G"

# 하이퍼파라미터
BATCH_SIZE=32
LR=1e-5
EPOCHS=100
STRIDE=512
SEQ_LEN=512
ALPHA=1.0      # prediction loss 가중치 (α)
BETA=0.002     # sigreg loss 가중치 (β); Epoch1 기준 P:S ≈ 1:1.5

# 아키텍처 설정: TILING (unsqueeze -> Linear(1, 512) -> in_chans=C)
ARCH="tiling"
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
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers 4
