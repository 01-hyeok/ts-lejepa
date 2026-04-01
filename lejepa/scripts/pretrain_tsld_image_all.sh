#!/bin/bash
# 앞의 학습이 끝나면 다음 줄로 넘어갑니다.
./scripts/pretrain_tsld_basic.sh > ./logs/pretrain_tsld_basic.log 2>&1
./scripts/pretrain_tsld_tiling.sh > ./logs/pretrain_tsld_tiling.log 2>&1
./scripts/pretrain_tsld_tiling_ci.sh > ./logs/pretrain_tsld_tiling_ci.log 2>&1
./scripts/pretrain_tsld_tivit_indep.sh > ./logs/pretrain_tsld_tivit_indep.log 2>&1
./scripts/pretrain_tsld_timevlm.sh > ./logs/pretrain_tsld_timevlm.log 2>&1
./scripts/pretrain_tsld_conv.sh > ./logs/pretrain_tsld_conv.log 2>&1