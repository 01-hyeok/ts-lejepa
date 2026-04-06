#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==========================================
# LeJEPA Classification - UCR Univariate Datasets
# Pretrained on: TSLD (C=1, 단변량)
# Target: UCR 10 Univariate Datasets (C=1)
# ==========================================

PRETRAIN_DATA="tsld"
ARCH="timesnet"
CHECKPOINT_TYPE="total"

BATCH_SIZE=16
LR=1e-03
NUM_EPOCHS=50
SEQ_LEN=0

# UCR 단변량 대표 데이터셋 10개 선정
# 기준: 샘플 수가 너무 적지 않고 (~50+), C=1 단변량
DATASETS=(
    "ECG200"
    "ECG5000"
    "FordA"
    "FordB"
)

echo "******************************************"
echo " Running UCR Classification for: ${ARCH}"
echo " Total Datasets: ${#DATASETS[@]}"
echo "******************************************"

PRETRAIN_PATH_SAVE="./checkpoints/pretrain/pretrain_${PRETRAIN_DATA}_${ARCH}/${ARCH}/lejepa_best_${CHECKPOINT_TYPE}_${PRETRAIN_DATA}.pt"

if [ ! -f "$PRETRAIN_PATH_SAVE" ]; then
    echo "⚠️ Warning: Pretrained checkpoint not found at $PRETRAIN_PATH_SAVE"
fi

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=========================================="
    echo " Processing Target Dataset: ${DATASET_NAME}"
    echo "------------------------------------------"

    DATA_ROOT="../Dataset/Time-Series-Library_dataset/UCR/${DATASET_NAME}"

    python -u run_classification.py \
        --arch ${ARCH} \
        --pretrain_dataset ${PRETRAIN_DATA} \
        --data_root "${DATA_ROOT}" \
        --dataset_name "${DATASET_NAME}" \
        --pretrain_path "${PRETRAIN_PATH_SAVE}" \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --epochs ${NUM_EPOCHS} \
        --seq_len ${SEQ_LEN} \
        --num_workers 0

    if [ $? -eq 0 ]; then
        echo "✓ ${DATASET_NAME} Classification 테스트 완료!"
    else
        echo "✗ ${DATASET_NAME} Classification 테스트 실패!"
    fi
done

echo "=========================================="
echo "모든 UCR Classification 스크립트 실행이 종료되었습니다."
echo "=========================================="
