#!/bin/bash

#python3.9 -c "import torch;device = 'cuda' if torch.cuda.is_available() else 'cpu';print(device);print('Cuda version with torch: {}'.format(torch.version.cuda))"
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib"
export PYTHONPATH="/jmain02/home/J2AD014/mtc03/lla98-mtc03/.conda/envs/ffcv/lib/python3.9/site-packages"

python3.9 -c "import os; print(os.environ)"

python3.9 extraction_external.py -f $1 -m $2 -t $3 -s $4 -d $5 --RF_level $6 --input_resolution $7 --num_workers $8  --batch_size $9 --save_path "${10}" --name "${11}"