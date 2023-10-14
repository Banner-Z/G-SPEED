
export CUDA_VISIBLE_DEVICES=0

NUM_CLUSTERS=10;
DATASET=wiki;
DATA_DIR=path2data;
KMEANS_DIR=path2output;

python scripts/train_sbert_kmeanspp_cluster.py \
--data-dir ${DATA_DIR} \
--num-clusters ${NUM_CLUSTERS} \
--output-dir ${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/ \
--do_svd True \
--n_init 16 \
