# Time Series Representation Models (TSRM)
This repository contains the implementation for the TSRM including all experimental setups.

## Requirements:
You need to install all packages provided in requirements.txt.

```
pip install -r requirements.txt
```

## Data:
All pre-processed benchmark datasets can be retrieved from [Google drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy), provided by [Wu et al](https://github.com/thuml/Autoformer). 
To run the experiments, you have to unzip them into the folder data_dir.

## Experiments:
All relevant experiment details can be found in the package "experiments". The folder "configs" contains the configurations for all experiments included in our article.
To run all experiments, you can execute the module "run.py" in the package "experiments" (set working directory and PYTHONPATH to the root folder).


