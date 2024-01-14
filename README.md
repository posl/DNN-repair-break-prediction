# DNN-repair-break-prediction
This is the replication package of our work of repairs and breaks prediction for DNNs.

We have tested to work with with AWS EC2 p2.xlarge instance (Ubuntu 18.04.6 LTS) and M1 Macbook Pro BigSur version 11.2.3.

# How to run

Current directory after clone this repository:

```bash
pwd
# /XXXX/YYYY/DNN-repair-break-prediction
```

## 0. Download datasets
Download the directory from google drive of the following link and put it under `DNN-repair-break-prediction/`: https://drive.google.com/drive/folders/1Z1W3eK2UYpP_bz2PLO-MhaT6TGvb2hKr?usp=drive_link

You can see the directory named `data`.
This directory has a subdirectory for each dataset used in the experiment and contains raw and preprocessed data.


## 1. Build docker container
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

The following commands are intended to be executed from a shell within this container.

## 2. Train target DNNs
Run the following bash file to train the DNN on the target dataset and obtain its performance.
```shell
bash shell/run_build_model.sh
``` 

It should be very time-consuming to train a model on all the datasets listed in `run_build_model.sh`.
Therefore, you can comment out some of the contents of `run_build_model.sh` to run it only on specific datasets.

## 3. Apply repair methods
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

## 4. Build repairs and breaks prediction models
