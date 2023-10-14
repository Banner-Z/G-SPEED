export CUDA_VISIBLE_DEVICES=0

DATA_DIR=path2training_data;
MODEL_PATH=path2model;
TAGGING_FILE=wnc_input_tagging.json;
OUT_DIR=path2output;
TASK_NUM=5;
SUM_TASKS=6;
SPARSE_MODE=ffd
SPARSE_ENCDEC=True
GATE_TYPE=task_id
HEAD_NUM=6
SPARSE_LEVEL=sentence_level
GATE_TEMPERATURE=1

accelerate launch run_mt_edit_model.py \
  --model_name_or_path ${MODEL_PATH} \
  --tagging_file_list ${DATA_DIR}/train1-tagging.json ${DATA_DIR}/train2-tagging.json \
  --validation_tagging_file_list ${DATA_DIR}/valid1-tagging.json ${DATA_DIR}/valid2-tagging.json \
  --prediction_tagging_file ${TAGGING_FILE} \
  --output_dir ${OUT_DIR} \
  --pad_to_max_length \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 5 \
  --return_entity_level_metrics \
  --only_do_predict \
  --checkpointing_steps epoch \
  --sum_tasks ${SUM_TASKS} \
  --task_num ${TASK_NUM} \
  --sparse_mode ${SPARSE_MODE} \
  --sparse_encdec ${SPARSE_ENCDEC} \
  --gate_type ${GATE_TYPE} \
  --head_num ${HEAD_NUM} \
  --sparse_level ${SPARSE_LEVEL} \
  --gate_temperature ${GATE_TEMPERATURE} \
  --max_length 512 \

python convert_data2json.py \
  -i ${TAGGING_FILE} \
  --pred ${OUT_DIR}/predictions.txt \
  -o ${OUT_DIR}/inference.json \
  --func editeval \
  --mode generation

accelerate launch run_mt_edit_model.py \
  --model_name_or_path ${MODEL_PATH} \
  --generation_file_list ${DATA_DIR}/train2-generation.json \
  --validation_generation_file_list ${DATA_DIR}/valid2-generation.json \
  --prediction_generation_file ${OUT_DIR}/inference.json \
  --output_dir ${OUT_DIR} \
  --pad_to_max_length \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 5 \
  --return_entity_level_metrics \
  --only_do_predict \
  --checkpointing_steps epoch \
  --token_num_per_slot 4 \
  --sum_tasks ${SUM_TASKS} \
  --task_num ${TASK_NUM} \
  --ignore_mismatched_sizes \
  --sparse_mode ${SPARSE_MODE} \
  --sparse_encdec ${SPARSE_ENCDEC} \
  --gate_type ${GATE_TYPE} \
  --head_num ${HEAD_NUM} \
  --sparse_level ${SPARSE_LEVEL} \
  --gate_temperature ${GATE_TEMPERATURE} \
  --max_length 512 \
