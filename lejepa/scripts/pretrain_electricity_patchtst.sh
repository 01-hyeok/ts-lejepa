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
PATCH_SIZE=16  # 새로 추가된 1D 패치 크기 파라미터
REVIN=False      # RevIN(Instance Norm) 적용 여부: True / False
LOCAL_LEN=256  # Local View 크롭 길이 (타임스텝): 128 -> 256 으로 확장

# 아키텍처 설정: patchtst (PatchTST 스타일 1D पै치 인코더)
ARCH="patchtst"
SAVE_DIR="./checkpoints/pretrain/pretrain_${DATA}_${ARCH}"
LOG_DIR="./runs/pretrain_${DATA}_${ARCH}"

echo "🚀 LeJEPA [${ARCH^^}] Pretraining 시작: ${DATA} (Patch Size: ${PATCH_SIZE})"

python run_ts_lejepa.py \
    --arch ${ARCH} \
    --patch_size ${PATCH_SIZE} \
    --local_len ${LOCAL_LEN} \
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
    --use_revin ${REVIN} \
    --num_workers 4
