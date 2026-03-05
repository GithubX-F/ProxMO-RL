#!/bin/bash

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

export http_proxy=
export https_proxy=
CONDA_PATH="${HOME}/.cache/model_data/conda"
export PATH="$CONDA_PATH/bin:$PATH"
source "$CONDA_PATH/etc/profile.d/conda.sh"
eval "$($CONDA_PATH/bin/conda shell.bash hook)"
type conda
conda activate webshop2
echo "setup当前环境: ${CONDA_DEFAULT_ENV}"
which python
python --version

# Install Python Dependencies
# pip install -r requirements.txt;
# pip install --default-timeout=300 --retries=20 -r requirements.txt \
#   -i  \
#   --trusted-host mirrors.aliyun.com;

# conda config --set remote_read_timeout_secs 300
# conda config --set remote_max_retries 20
# conda config --set remote_connect_timeout_secs 300
# conda config --add channels 
# conda config --add channels 
# conda config --add channels 
# conda config --add channels 
# conda config --set show_channel_urls yes
# conda install mkl -y
# conda install -c conda-forge faiss-cpu -y
# conda install -c conda-forge openjdk=11 -y;

# Download dataset into `data` folder via `gdown` command
mkdir -p data;
cd data;
ln -s ${HOME}/.cache/model_data/data/webshop/items_shuffle_1000.json ./
ln -s ${HOME}/.cache/model_data/data/webshop/items_ins_v2_1000.json ./
ln -s ${HOME}/.cache/model_data/data/webshop/items_human_ins.json ./
ln -s ${HOME}/.cache/model_data/data/webshop/items_ins_v2.json ./
ln -s ${HOME}/.cache/model_data/data/webshop/items_shuffle.json ./

# gdown  # items_shuffle_1000 - product scraped info
# gdown  # items_ins_v2_1000 - product attributes
# gdown  # items_shuffle
# gdown  # items_ins_v2
# gdown  # items_human_ins
cd ..

# Download spaCy large NLP model
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_sm
# 检查并安装small模型
if ! pip show en-core-web-sm >/dev/null 2>&1; then
    echo "安装 en_core_web_sm..."
    pip install 
else
    echo "✓ en_core_web_sm 已安装，跳过"
fi

# 检查并安装large模型
if ! pip show en-core-web-lg >/dev/null 2>&1; then
    echo "安装 en_core_web_lg..."
    pip install 
else
    echo "✓ en_core_web_lg 已安装，跳过"
fi
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

