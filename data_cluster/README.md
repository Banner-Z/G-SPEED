# Data Cluster
```
bash train_kmeanspp.sh
```
You need to install [cuml](https://docs.rapids.ai/install). Because the amount of data is large, it is more efficient to train the K-Means model on GPU. At the same time, since the training results of K-Means will be greatly affected by initialization, it is often necessary to train the model multiple times or perform post-processing according to the actual situation to improve the quality of the data set.
