
# G-SPEED
The official repository of paper "G-SPEED: General SParse Efficient Editing MoDel".

![img](./gspeed.png)

## Set up enviroment
### Clone this repository
```bash
git clone https://github.com/Banner-Z/G-SPEED.git
cd G-SPEED
```
### Create a conda env
```bash
conda create -n gspeed
conda activate gspeed
```
Alternatively, we supply a `requirements.txt` file; which can be used to create a conda environment by `pip install -r requirements.txt`.

The necessary environments for G-SPEED include: `torch, transformers, datasets, evaluate, and accelerate`. You can build an environment that includes these packages to get started quickly.

If you want to perform data clustering yourself, you need to install [cuml](https://docs.rapids.ai/install), sklearn, and sentence_transformers. In addition, we use [EditEval](https://github.com/facebookresearch/EditEval) for evaluation. If you need to perform evaluation, you can refer to their method.

## G-SPEED Training and Evaluation

### Step 0: Download the dataset or collect it from scratch.
Our dataste is available at https://huggingface.co/datasets/zhk/wiki-edits.
Each piece of data includes three attributes: 'source', 'target', and 'intent'. There are a total of 506k pieces of data, including four intents: "Readability", "Neutralization", "Fluency", "Simplification".

If you need to obtain new clustering data, you can collect data according to ./data_collector, and then cluster the data according to ./data_cluster.

### Step 1: Annotate editing actions using dynamic programming.


### Step 2: Pre-training and additional fine-tuning.
Our model is available at https://huggingface.co/zhk/G-SPEED.

### Step 3: Inference and evaluation.


## Citation
Coming soon.
