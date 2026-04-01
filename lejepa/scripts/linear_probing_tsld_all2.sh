#!/bin/bash
# 앞의 학습이 끝나면 다음 줄로 넘어갑니다.
./scripts/linear_probing_tsld_tiling.sh > ./logs/linear_probing_tsld_tiling.log 2>&1
./scripts/linear_probing_tsld_tiling_ci.sh > ./logs/linear_probing_tsld_tiling_ci.log 2>&1
./scripts/linear_probing_tsld_tivit_indep.sh > ./logs/linear_probing_tsld_tivit_indep.log 2>&1
