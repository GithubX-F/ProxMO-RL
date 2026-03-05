#!/bin/bash

_shared_cache="${HOME}/.cache/model_data/data/alfworld"
_local_cache="$HOME/.cache/alfworld"

echo "===================================================="
echo "开始复制数据..."
echo "源: ${_shared_cache}"
echo "目标: ${_local_cache}"
echo "===================================================="

mkdir -p ${_local_cache}

rsync -a ${_shared_cache}/* ${_local_cache}/ && \
echo "" && \
echo "✅ 复制完成" && \
echo "" && \
echo "复制后的目录内容:" && \
ls -lh ${_local_cache}/ && \
echo "" && \
echo "目录大小:" && \
du -sh ${_local_cache} || \
echo "❌ 复制失败"