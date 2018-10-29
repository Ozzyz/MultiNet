# Usage: train.sh {GPU}

source activate py2
python train.py --hypes hypes/multinet3.json --gpus $1
