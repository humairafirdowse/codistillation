# CO-DISTILLATION
This repository contains all the code needed to reproduce the codistillation results presented in our paper. 


## Folder structure
This repository is organised as follows:
- `run.py`: Executable containg the code for the main algorithm (see function documentation).
- `helpers.py`: Module containing all the helpers function for loading/training/evaluation/etc.
- `models.py`: Module containing all the CNN models that are used.
- `data_maker.py`: Module containing all data pre processing functions. 
- `LICENSE`: MIT Licence.
- `requirements.txt`: Text file containing the required libraries to run the code.

## Requirements

To install requirements, install [Minconda](https://docs.conda.io/en/latest/miniconda.html), then create and activate a new conda environment with the commands:

```setup
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```


## Execution

To run an experiment, execute the following command (only an example, type `python run.py -h` for a detailed explanation of all the arguments):
```train
python run.py --n_clients 2 --dataset CELEBA  --model LeNet5 --feature_dim 84 --rounds 100
```