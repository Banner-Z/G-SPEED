export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=24

DATA_DIR=path2data;
accelerate launch --num_processes 4 --multi_gpu --main_process_port 21247 run_mt_edit_model.py \
  --model_name_or_path path2pretrained_model \
  --tagging_file_list ${DATA_DIR}/trainFLU-tagging.json ${DATA_DIR}/trainCLA-tagging.json ${DATA_DIR}/trainPAR-tagging.json ${DATA_DIR}/trainSIM-tagging.json ${DATA_DIR}/trainNEU-tagging.json \
  --validation_tagging_file_list ${DATA_DIR}/trainFLU-tagging.json ${DATA_DIR}/trainCLA-tagging.json ${DATA_DIR}/trainPAR-tagging.json ${DATA_DIR}/trainSIM-tagging.json ${DATA_DIR}/trainNEU-tagging.json \
  --generation_file_list ${DATA_DIR}/trainFLU-generation.json ${DATA_DIR}/trainCLA-generation.json ${DATA_DIR}/trainPAR-generation.json ${DATA_DIR}/trainSIM-generation.json ${DATA_DIR}/trainNEU-generation.json \
  --validation_generation_file_list ${DATA_DIR}/trainFLU-generation.json ${DATA_DIR}/trainCLA-generation.json ${DATA_DIR}/trainPAR-generation.json ${DATA_DIR}/trainSIM-generation.json ${DATA_DIR}/trainNEU-generation.json \
  --output_dir output_dir \
  --pad_to_max_length \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 3 \
  --return_entity_level_metrics \
  --do_train \
  --do_eval \
  --checkpointing_steps epoch \
  --sum_tasks 6 \
  --sparse_mode ffd \
  --sparse_encdec True \
  --gate_type task_id \
  --head_num 6 \
  --sparse_level sentence_level \
  --gate_temperature 0.7 \
  --learning_rate 5e-5 \
  --max_grad_norm 1 \
  --is_additional_finetune True \
  --copy_ids 2 0 0 3 1 \
  --is_freeze True \

