#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==========================================
# LeJEPA Linear Probing
# Arch: BASIC (C -> 512)
# ==========================================

ARCH="basic"
PRETRAIN_DATA="tsld"
PRETRAIN_PATH="./outputs/pretrain/${ARCH}/lejepa_best_${PRETRAIN_DATA}.pt"

TARGET_DATASETS=("tsld") # TSLD 자체 평가 또는 타 데이터셋 전이
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE=32
LR=1e-4
NUM_EPOCHS=30

for DATA in "${TARGET_DATASETS[@]}"; do
    # Path logic for TSLD or others
    if [ "${DATA}" = "tsld" ]; then
        DATA_PATH="/data/pjh_workspace/Dataset/TSLD-1G"
    else
        DATA_PATH="/data/pjh_workspace/Dataset/Time-Series-Library_dataset/${DATA}/${DATA}.csv"
    fi
    
    LOG_DIR="./outputs/linear_probing/${ARCH}_${PRETRAIN_DATA}_to_${DATA}"
    
    for PRED_LEN in "${PRED_LENGTHS[@]}"; do
        python run_linear_probing_lejepa.py \
            --arch ${ARCH} \
            --pretrain_dataset ${PRETRAIN_DATA} \
            --target_dataset ${DATA} \
            --data_path "${DATA_PATH}" \
            --pretrain_path "${PRETRAIN_PATH}" \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --epochs ${NUM_EPOCHS} \
            --pred_len ${PRED_LEN} \
            --log_dir "${LOG_DIR}"
    done
done
