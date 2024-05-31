# DNN-repair-break-prediction
This is the replication package of our study of repairs and breaks prediction for DNNs.

This repository contains all the source code and datasets used in our study.

We have tested to work with with AWS EC2 p2.xlarge instance (Ubuntu 18.04.6 LTS) and M1 Macbook Pro BigSur version 11.2.3.

# How to run

Current directory after cloning this repository:

```bash
pwd
# /XXXX/YYYY/DNN-repair-break-prediction
```

## Step 0. Download datasets
Download the directory from google drive of the following link and put it under `DNN-repair-break-prediction/`: https://drive.google.com/drive/folders/1Z1W3eK2UYpP_bz2PLO-MhaT6TGvb2hKr?usp=drive_link

You can see a directory named `data`.
This directory has a subdirectory for each dataset used in the experiment and contains raw and preprocessed data.

This link also contains a directory named `repairs-breaks-dataset`.
This directory has a subdirectory for each repair method.
These subdirectories contain a csv of the repair history for each repair method.
The `raw_data` contains the values of the explanatory variables as they are, and the `preprocessed_data` contains the values after the prescribed preprocessing of the explanatory variables.
These datasets can be obtained at the end of Step 3 described below, but is cut out on the drive for easy review.


## Step 1. Build docker container
When use only CPU (note: slow to run the experiment related to DNN training and inference):
```shell
docker compose up -d
```

When use GPU:
```shell
docker compose -f docker-compose.gpu.yml up -d
``` 

At this point, a container named `NN-repair-break` should be running.

After that, enter the shell of the docker container with the following command.
```shell
docker exec -it NN-repair-break bash
```

The current directory after entering the shell should be below.
```shell
pwd
# /src
```
The following commands are intended to be executed from the above path.

## Step 2. Train target DNNs
Run the following bash file to train the DNN on the target dataset and obtain its performance.
```shell
bash shell/run_build_model.sh
``` 

It should be very time-consuming to train a model on all the datasets listed in `run_build_model.sh`.
Therefore, you can comment out some of the contents of `run_build_model.sh` to run it only on specific datasets.

## Step 3. Apply repair methods
This is the most time-consuming part of this repository.

CARE
```shell
bash shell/run_care_repair.sh
```

Apricot
```shell
bash shell/run_apricot_repair.sh
```

Arachne
```shell
bash shell/run_arachne_repair.sh
```

## Step 4. Build repairs and breaks prediction models
Based on the history of repair in the previous step, build repairs and breaks prediction models.

First, preprocess the repairs and breaks datasets by running the following python file:
```shell
python shell/run_preprocess_repair_break_dataset.py
```

Then, build repairs and breaks prediction models for each dataset and repair method:
```shell
python shell/run_build_repair_break_model.py
```

## Step 5. Run experiments for RQs
Change current directory to run the experiments for RQs.
```shell
cd src/
pwd
# /src/src
```

### RQ1
```shell
python utest.py repair
python utest.py break
```

### RQ2
```shell
python plot_bar_repair_break_pred.py
```

### RQ3
```shell
python measure_perf_selection.py
```

### RQ4
```shell
python ../shell/run_transfer_methods.py
python plot_transfer_line.py
```