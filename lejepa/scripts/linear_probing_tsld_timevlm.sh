#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# ==========================================
# LeJEPA Linear Probing - Transfer Learning
# Pretrained on: Electricity
# Target datasets: ETTm1, ETTm2, Weather
# ==========================================

PRETRAIN_DATA="tsld"
ARCH="timevlm"
PRETRAIN_PATH_SAVE="./checkpoints/pretrain_${PRETRAIN_DATA}_${ARCH}/${ARCH}/lejepa_best_${PRETRAIN_DATA}.pt"

if [ ! -f "$PRETRAIN_PATH_SAVE" ]; then
    echo "⚠️ Error: Pretrained checkpoint not found at $PRETRAIN_PATH_SAVE"
    echo "Please pretrain the model first."
    exit 1
fi

TARGET_DATASETS=("ETTm1" "ETTm2" "weather")
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE=32
LR=1e-04
NUM_EPOCHS=20

for DATA in "${TARGET_DATASETS[@]}"; do
    echo "=========================================="
    echo " Processing Target: ${DATA}"
    echo "=========================================="
    
    # Dataset path resolution
    if [ "${DATA}" = "ETTm1" ] || [ "${DATA}" = "ETTm2" ]; then
        DATA_DIR="ETT-small"
    else
        DATA_DIR="${DATA}"
    fi
    DATA_PATH="../Dataset/Time-Series-Library_dataset/${DATA_DIR}/${DATA}.csv"
    
    LOG_DIR="./checkpoints/linear_probing/LeJEPA_${PRETRAIN_DATA}_to_${DATA}_${ARCH}"
    mkdir -p ${LOG_DIR}
    
    for PRED_LEN in "${PRED_LENGTHS[@]}"; do
        echo "----------------------------------------"
        echo " Prediction Length: ${PRED_LEN}"
        echo "----------------------------------------"
        
        python -u run_linear_probing_lejepa.py \
            --arch ${ARCH} \
            --pretrain_dataset ${PRETRAIN_DATA} \
            --target_dataset ${DATA} \
            --data_path "${DATA_PATH}" \
            --pretrain_path "${PRETRAIN_PATH_SAVE}" \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --epochs ${NUM_EPOCHS} \
            --pred_len ${PRED_LEN} \
            --num_workers 0 \
            --log_dir "${LOG_DIR}"
            
        if [ $? -eq 0 ]; then
            echo "✓ ${DATA} (pred_len=${PRED_LEN}) 완료!"
        else
            echo "✗ ${DATA} (pred_len=${PRED_LEN}) 실패!"
        fi
    done
done

echo "=========================================="
echo "모든 Linear Probing 스크립트 실행이 완료되었습니다."
echo "결과는 ./checkpoints/linear_probing/results_summary.txt 에서 확인하세요."
echo "=========================================="
