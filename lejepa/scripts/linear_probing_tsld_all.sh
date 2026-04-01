#!/bin/bash
# 앞의 학습이 끝나면 다음 줄로 넘어갑니다.
./scripts/linear_probing_tsld_patchtst.sh > ./logs/linear_probing_tsld_patchtst.log 2>&1
./scripts/linear_probing_tsld_utica.sh > ./logs/linear_probing_tsld_utica.log 2>&1
./scripts/linear_probing_tsld_basic.sh > ./logs/linear_probing_tsld_basic.log 2>&1