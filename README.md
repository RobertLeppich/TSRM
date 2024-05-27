# Time Series Representation Models (TSRM)
This repository contains the implementation for the TSRM including all experimental setups.

## Requirements:
You need to install all packages provided in requirements.txt.

```
pip install -r requirements.txt
```

## Data:
We provide all pre-processed benchmark datasets in an (anonymous) [Google drive](https://drive.google.com/drive/folders/1Sw6LClDcYhy5byltrezagiap9a5sGIfH).
To run the experiments, you have to unzip them into the folder data/data_dir.

Additional we provide all code utilized to pre-process the original data in the package data/preprocessing.

## Experiments:
All relevant experiment details can be found in the package "experiments". The folder "configs" contains the configurations for all experiments included in our article.
To run all experiments, you can execute the module "run.py" in the package "experiments" (set working directory and PYTHONPATH to the root folder). All experiments will train a pre-train model first and fine-tune it afterward.


