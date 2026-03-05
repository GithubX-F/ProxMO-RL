#!/usr/bin/env bash
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" ""
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" ""
fi
