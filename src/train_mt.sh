export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=24

DATA_DIR=path2data;
accelerate launch --num_processes 8 --multi_gpu --main_process_port 11247 run_mt_edit_model.py \
  --model_name_or_path bert-base-cased \
  --tagging_file_list ${DATA_DIR}/train1-tagging.json ${DATA_DIR}/train2-tagging.json ${DATA_DIR}/train3-tagging.json ${DATA_DIR}/train4-tagging.json \
  --validation_tagging_file_list ${DATA_DIR}/valid1-tagging.json ${DATA_DIR}/valid2-tagging.json ${DATA_DIR}/valid3-tagging.json ${DATA_DIR}/valid4-tagging.json \
  --generation_file_list ${DATA_DIR}/train1-generation.json ${DATA_DIR}/train2-generation.json ${DATA_DIR}/train3-generation.json ${DATA_DIR}/train4-generation.json \
  --validation_generation_file_list ${DATA_DIR}/valid1-generation.json ${DATA_DIR}/valid2-generation.json ${DATA_DIR}/valid3-generation.json ${DATA_DIR}/valid4-generation.json \
  --output_dir path2output \
  --pad_to_max_length \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 10 \
  --return_entity_level_metrics \
  --do_train \
  --do_eval \
  --checkpointing_steps epoch \
  --sum_tasks 4 \
  --sparse_mode ffd \
  --sparse_encdec True \
  --gate_type task_id \
  --head_num 4 \
  --sparse_level sentence_level \
  --gate_temperature 0.7 \
  --learning_rate 5e-5 \
  --max_grad_norm 1 \
