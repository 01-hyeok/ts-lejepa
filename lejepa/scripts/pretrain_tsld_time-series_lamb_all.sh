#!/bin/bash
# 앞의 학습이 끝나면 다음 줄로 넘어갑니다.
./scripts/pretrain_tsld_patchtst_lamb.sh > ./logs/pretrain_tsld_patchtst_lamb.log 2>&1
./scripts/pretrain_tsld_utica_lamb.sh > ./logs/pretrain_tsld_utica_lamb.log 2>&1