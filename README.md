# 4th place solution (tattaka's part)

## Environment
Use [Kaggle Docker](https://console.cloud.google.com/gcr/images/kaggle-gpu-images/GLOBAL/python).

## Usage
0. Place competition data in the `input` directory
1. make bp_matrix
    ```bash
    $ cd src/scripts
    $ python make_structure_feat_train.py
    $ python make_structure_feat_test.py
    ```
    Place contrafold bpps and unzip in in the `input` directory

2. Training
    ```bash
    $ cd src/exp064 && sh train.sh
    $ cd src/exp070 && sh train.sh
    $ cd src/exp071 && sh train.sh 
    $ cd src/exp072 && sh train.sh
    ```
## License
MIT
