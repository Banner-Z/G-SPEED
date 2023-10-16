export CUDA_VISIBLE_DEVICES=0

DATA_DIR=path2training_data;
ENC_MODEL_PATH=path2encoder;
DEC_MODEL_PATH=path2decoder; # ENC_MODEL_PATH and DEC_MODEL_PATH may be the same.

INPUT_FILE_LIST=() # a list of json files, which will be used when testing on EditEval. The Structure of Files : {'id': xxx, 'input':xxxxx}

TAGGING_FILE_LIST=() # a list of json files. You can find the details of tagging files in dp_edit_actions.

OUTPUT_PATH_LIST=()

TASK_NAME_LIST=(jfleg iterater_fluency iterater_clarity iterater_coherence stsb_multi_mt turk asset wnc)
INTENT_LIST=(fluency clarity coherence paraphrasing simplification neutral)

TASK_NUM_LIST=(0 0 1 2 3 4 4 5)
SUM_TASKS=6;
SPARSE_MODE=ffd
SPARSE_ENCDEC=True
GATE_TYPE=task_id
HEAD_NUM=6
SPARSE_LEVEL=sentence_level
GATE_TEMPERATURE=0.7

for ((i=0;i<${#TASK_NAME_LIST[@]};i++))
do
    if test ${TASK_NUM_LIST[$i]} -eq -1
    then
        continue
    else
        echo 'begin test'
    fi

    index=${TASK_NUM_LIST[$i]}
    OUT_PATH=${OUTPUT_PATH_LIST[$i]}/${INTENT_LIST[$index]}/;
    accelerate launch run_mt_edit_model.py \
        --model_name_or_path ${ENC_MODEL_PATH} \
        --tagging_file_list ${DATA_DIR}/train1-tagging.json ${DATA_DIR}/train2-tagging.json ${DATA_DIR}/train3-tagging.json ${DATA_DIR}/train4-tagging.json \
        --validation_tagging_file_list ${DATA_DIR}/valid1-tagging.json ${DATA_DIR}/valid2-tagging.json ${DATA_DIR}/valid3-tagging.json ${DATA_DIR}/valid4-tagging.json \
        --prediction_tagging_file ${TAGGING_FILE_LIST[$i]} \
        --output_dir ${OUT_PATH} \
        --pad_to_max_length \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 5 \
        --return_entity_level_metrics \
        --only_do_predict \
        --checkpointing_steps epoch \
        --sum_tasks ${SUM_TASKS} \
        --task_num ${TASK_NUM_LIST[$i]} \
        --sparse_mode ${SPARSE_MODE} \
        --sparse_encdec ${SPARSE_ENCDEC} \
        --gate_type ${GATE_TYPE} \
        --head_num ${HEAD_NUM} \
        --sparse_level ${SPARSE_LEVEL} \
        --gate_temperature ${GATE_TEMPERATURE}
    
    python convert_data2json.py \
        -i ${TAGGING_FILE_LIST[$i]} \
        --pred ${OUT_PATH}/predictions.txt \
        -o ${OUT_PATH}/inference.json \
        --func editeval \
        --mode generation

    accelerate launch run_mt_edit_model.py \
        --model_name_or_path ${DEC_MODEL_PATH}  \
        --generation_file_list ${DATA_DIR}/train1-generation.json ${DATA_DIR}/train2-generation.json ${DATA_DIR}/train3-generation.json ${DATA_DIR}/train4-generation.json \
        --validation_generation_file_list ${DATA_DIR}/valid1-generation.json ${DATA_DIR}/valid2-generation.json ${DATA_DIR}/valid3-generation.json ${DATA_DIR}/valid4-generation.json \
        --prediction_generation_file ${OUT_PATH}/inference.json \
        --output_dir ${OUT_PATH} \
        --pad_to_max_length \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 5 \
        --return_entity_level_metrics \
        --only_do_predict \
        --checkpointing_steps epoch \
        --token_num_per_slot 4 \
        --sum_tasks ${SUM_TASKS} \
        --task_num ${TASK_NUM_LIST[$i]} \
        --ignore_mismatched_sizes \
        --sparse_mode ${SPARSE_MODE} \
        --sparse_encdec ${SPARSE_ENCDEC} \
        --gate_type ${GATE_TYPE} \
        --head_num ${HEAD_NUM} \
        --sparse_level ${SPARSE_LEVEL} \
        --gate_temperature ${GATE_TEMPERATURE}

done

# There is a demo of using EditEval Benchmark to evaluate model outputs. If you are not using this benchmark for evaluation, ignore the code below. 
source /opt/conda/etc/profile.d/conda.sh
conda activate editeval

for ((i=0;i<${#TASK_NAME_LIST[@]};i++))
do
    if test ${TASK_NUM_LIST[$i]} -eq -1
    then
        continue
    fi
    index=${TASK_NUM_LIST[$i]}
    OUT_PATH=${OUTPUT_PATH_LIST[$i]}/+${INTENT_LIST[$index]}/;
	
	cd the_path2src

	python convert_data2json.py \
		-i ${INPUT_FILE_LIST[$i]} \
		--pred ${OUT_PATH}/final_results.txt \
		-o ${OUT_PATH}/final_results.jsonl \
		--func editeval \
		--mode results2json

	cd the_path2editeval

	python main.py \
		--dataset_name ${TASK_NAME_LIST[$i]} \
		--prediction_file ${OUT_PATH}/final_results.jsonl \

    cd the_path2src

    python convert_data2json.py \
    -i ${OUT_PATH}/final_results.txt \
    -o ${OUT_PATH}/input_tagging.json \
    --func editeval \
    --mode tagging
    
done
