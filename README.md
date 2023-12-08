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
    * https://www.kaggle.com/code/tattaka/contrafold-test-bpps-1000000
    * https://www.kaggle.com/code/tattaka/contrafold-test-bpps-500000-1000000
    * https://www.kaggle.com/code/tattaka/contrafold-test-bpps-0-500000
    * https://www.kaggle.com/code/tattaka/contrafold-train-bpps-600000
    * https://www.kaggle.com/code/tattaka/contrafold-train-bpps-300000-600000
    * https://www.kaggle.com/code/tattaka/contrafold-train-bpps-0-300000 

2. Training
    ```bash
    $ cd src/exp064 && sh train.sh
    $ cd src/exp070 && sh train.sh
    $ cd src/exp071 && sh train.sh 
    $ cd src/exp072 && sh train.sh
    ```
## License
MIT
