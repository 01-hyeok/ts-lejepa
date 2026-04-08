#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==========================================
# LeJEPA Classification - Transfer Learning
# Pretrained on: TSLD
# Target datasets: UEA 10 Multivariate Datasets
# ==========================================

PRETRAIN_DATA="tsld"
ARCH="patchtst"
CHECKPOINT_TYPE="total" # or "pred"

BATCH_SIZE=16
LR=1e-04
NUM_EPOCHS=30
SEQ_LEN=0 # 0 means auto-detect from dataset

# 평가를 진행할 10개의 UEA 데이터셋 리스트
DATASETS=(
    "EthanolConcentration"
    "FaceDetection"
    "Handwriting"
    "Heartbeat"
    "JapaneseVowels"
    "PEMS-SF"
    "SelfRegulationSCP1"
    "SelfRegulationSCP2"
    "SpokenArabicDigits"
    "UWaveGestureLibrary"
)

echo "******************************************"
echo " Running Classification for: ${ARCH}"
echo " Total Datasets: ${#DATASETS[@]}"
echo "******************************************"

# 불러올 체크포인트 경로 설정 (LeJEPA Pretraining 저장 규칙 따름)
PRETRAIN_PATH_SAVE="./checkpoints/pretrain/pretrain_${PRETRAIN_DATA}_${ARCH}/${ARCH}/lejepa_best_${CHECKPOINT_TYPE}_${PRETRAIN_DATA}.pt"

if [ ! -f "$PRETRAIN_PATH_SAVE" ]; then
    echo "⚠️ Warning: Pretrained checkpoint not found at $PRETRAIN_PATH_SAVE"
    echo "진행 전 Pretrain 경로가 유효한지 한 번 확인해 주세요!"
    # 테스트 목적이라면 주석 처리하고 진행할 수 있습니다.
fi

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=========================================="
    echo " Processing Target Dataset: ${DATASET_NAME}"
    echo "------------------------------------------"

    # 이전에 선생님께서 변경해주신 경로 구조 반영
    DATA_ROOT="../Dataset/Time-Series-Library_dataset/UEA/${DATASET_NAME}"
    
    # 로그 디렉토리 (Classification 고유 구조)
    LOG_DIR="./checkpoints/classification/classification_${PRETRAIN_DATA}_to_${DATASET_NAME}_${ARCH}_${CHECKPOINT_TYPE}"
    mkdir -p "${LOG_DIR}"

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
echo "모든 Classification 스크립트 실행이 종료되었습니다."
echo "결과는 ./checkpoints/classification/ 폴더 하위 각 실험의 results_summary.txt 에서 확인하세요."
echo "=========================================="
