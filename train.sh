# Usage: train.sh {GPU}

source activate py2
echo "Training on gpu $1"
python train.py --hypes hypes/multinet3.json --gpus $1
