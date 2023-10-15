# G-SPEED
The official repository of paper "G-SPEED: General SParse Efficient Editing MoDel".

## Set up enviroment
### Clone this repository
```bash
git clone https://github.com/kernelmachine/cbtm.git
cd cbtm
```
### Create a conda env
```bash
conda create -n gspeed
conda activate gspeed
```
Alternatively, we supply an `requirements.txt` file; which can be used to create a conda environment by `pip install -r requirements.txt`.

The necessary environments for G-SPEED include: `torch, transformers, datasets, evaluate, and accelerate`. You can build an environment that includes these packages to get started quickly.

If you want to perform data clustering yourself, you need to install [cuml](https://docs.rapids.ai/install), sklearn, and sentence_transformers. In addition, we use [EditEval](https://github.com/facebookresearch/EditEval) for evaluation. If you need to perform evaluation, you can refer to their method.

## G-SPEED Training and Evaluation

### Step 0: Download the dataset or collect it from scratch.


### Step 1: Annotate editing actions using dynamic programming.


### Step 2: Pre-training and additional fine-tuning.


### Step 3: Inference and evaluation.


## Citation
