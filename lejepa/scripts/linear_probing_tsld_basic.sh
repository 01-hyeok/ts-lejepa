#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==========================================
# LeJEPA Linear Probing - Transfer Learning
# Pretrained on: Electricity
# Target datasets: ETTm1, ETTm2, Weather
# ==========================================

PRETRAIN_DATA="tsld"
ARCH="basic"

# 체크포인트 타입 정의 (total, pred 둘 다 수행)
CHECKPOINT_TYPES=("total" "pred")

TARGET_DATASETS=("ETTm1" "ETTm2" "weather")
PRED_LENGTHS=(96 192 336 720)

BATCH_SIZE=32
LR=1e-04
NUM_EPOCHS=20

for CHECKPOINT_TYPE in "${CHECKPOINT_TYPES[@]}"; do
    echo "******************************************"
    echo " Running Linear Probing for: ${CHECKPOINT_TYPE}"
    echo "******************************************"
    
    # 불러올 체크포인트 경로 설정
    PRETRAIN_PATH_SAVE="./checkpoints/pretrain/pretrain_${PRETRAIN_DATA}_${ARCH}/${ARCH}/lejepa_best_${CHECKPOINT_TYPE}_${PRETRAIN_DATA}.pt"

    if [ ! -f "$PRETRAIN_PATH_SAVE" ]; then
        echo "⚠️ Warning: Pretrained checkpoint not found at $PRETRAIN_PATH_SAVE"
        echo "Skipping ${CHECKPOINT_TYPE}..."
        continue
    fi

    for DATA in "${TARGET_DATASETS[@]}"; do
        echo "=========================================="
        echo " Processing Target: ${DATA} (Criterion: ${CHECKPOINT_TYPE})"
        echo "=========================================="
        
        # Dataset path resolution
        if [ "${DATA}" = "ETTm1" ] || [ "${DATA}" = "ETTm2" ]; then
            DATA_DIR="ETT-small"
        else
            DATA_DIR="${DATA}"
        fi
        DATA_PATH="../Dataset/Time-Series-Library_dataset/${DATA_DIR}/${DATA}.csv"
        
        # 로그 디렉토리에 체크포인트 타입 명시
        LOG_DIR="./checkpoints/linear_probing/LeJEPA_${PRETRAIN_DATA}_to_${DATA}_${ARCH}_${CHECKPOINT_TYPE}"
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
                --use_revin True \
                --num_workers 0 \
                --log_dir "${LOG_DIR}"
                
            if [ $? -eq 0 ]; then
                echo "✓ ${DATA} (${CHECKPOINT_TYPE}, pred_len=${PRED_LEN}) 완료!"
            else
                echo "✗ ${DATA} (${CHECKPOINT_TYPE}, pred_len=${PRED_LEN}) 실패!"
            fi
        done
    done
done

echo "=========================================="
echo "모든 Linear Probing 스크립트 실행이 완료되었습니다."
echo "결과는 ./checkpoints/linear_probing/ 에서 확인하세요."
echo "=========================================="
