#!/bin/bash

# GPU 설정
export CUDA_VISIBLE_DEVICES=0

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
BETA=0.0005       # sigreg loss 가중치 (β); Loss 값이 아닌 모델을 훈련시키는 힘(Gradient)을 1:1로 맞추기 위해 가중치를 동일하게 부여함
PATCH_SIZE=16  # 새로 추가된 1D 패치 크기 파라미터
REVIN=False      # RevIN(Instance Norm) 적용 여부: True / False
LOCAL_LEN=256

# 아키텍처 설정: patchtst (PatchTST 스타일 1D पै치 인코더)
ARCH="patchtst"
SAVE_DIR="./checkpoints/pretrain/pretrain_${DATA}_${ARCH}_lamb"
LOG_DIR="./runs/pretrain_${DATA}_${ARCH}_lamb"

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
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --use_revin ${REVIN} \
    --num_workers 4
