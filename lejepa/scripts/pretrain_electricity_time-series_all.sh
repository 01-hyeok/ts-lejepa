#!/bin/bash
# 앞의 학습이 끝나면 다음 줄로 넘어갑니다.
./scripts/pretrain_electricity_patchtst.sh > ./logs/pretrain_electricity_patchtst.log 2>&1
./scripts/pretrain_electricity_utica.sh > ./logs/pretrain_electricity_utica.log 2>&1
./scripts/pretrain_electricity_conv.sh > ./logs/pretrain_electricity_conv.log 2>&1
./scripts/pretrain_electricity_timevlm.sh > ./logs/pretrain_electricity_timevlm.log 2>&1