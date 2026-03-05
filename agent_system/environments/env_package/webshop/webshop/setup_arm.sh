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

# Install Python Dependencies
conda install spacy
pip install -r requirements_arm.txt;

# Install Environment Dependencies via `conda`
conda install -c pytorch faiss-cpu;
conda install -c conda-forge openjdk=11;

# Download dataset into `data` folder via `gdown` command
mkdir -p data;
cd data;
if [ "$data" == "small" ]; then
  gdown  # items_shuffle_1000 - product scraped info
  gdown  # items_ins_v2_1000 - product attributes
elif [ "$data" == "all" ]; then
  gdown  # items_shuffle
  gdown  # items_ins_v2
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
gdown  # items_human_ins
cd ..

# Download spaCy large NLP model
python -m spacy download en_core_web_sm

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

# Create logging folder + samples of log data
get_human_trajs () {
  PYCMD=$(cat <<EOF
import gdown
url=""
gdown.download_folder(url, quiet=True, remaining_ok=True)
EOF
  )
  python -c "$PYCMD"
}
mkdir -p user_session_logs/
cd user_session_logs/
echo "Downloading 50 example human trajectories..."
get_human_trajs
echo "Downloading example trajectories complete"
cd ..