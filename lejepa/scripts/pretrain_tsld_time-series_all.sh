#!/bin/bash
# 앞의 학습이 끝나면 다음 줄로 넘어갑니다.
./scripts/pretrain_tsld_patchtst.sh > ./logs/pretrain_tsld_patchtst.log 2>&1
./scripts/pretrain_tsld_utica.sh > ./logs/pretrain_tsld_utica.log 2>&1