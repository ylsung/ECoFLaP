# This scripts trains Adapters method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. For all datasets we tried
# with the adapter's bottleneck size of `task_reduction_factor`=[32, 16, 8], and report the 
# results on the test set for the model performing the best on the validation set.

folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

source scripts/env.sh

file_name=full_finetuning

model_name_or_path=t5-small

# --model_name_or_path /playpen-ssd/ylsung/data/outputs/${model_name_or_path} \
# --tokenizer_name /playpen-ssd/ylsung/data/outputs/${model_name_or_path} \

output_path=${home_path}/data/outputs/${file_name}_2
for seed in 0
do

for task in "winogrande_debiased" # "squad_v2" "superglue-record" "superglue-multirc" "superglue-boolq" "superglue-cb" "superglue-multirc" "superglue-wic" "superglue-copa"

do
    rm -rf outputs/${file_name}/
    
    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py \
        --training_config_file configs/${file_name}.json \
        --task_name $task --eval_dataset_name $task --test_dataset_name $task \
        --seed ${seed} \
        --num_train_epochs ${num_epochs[$task]} \
        --max_source_length ${max_source_length[$task]} \
        --per_device_train_batch_size ${batch_size[$task]} \
        --per_device_eval_batch_size ${batch_size[$task]} \
        --output_dir ${output_path} \
        --model_name_or_path ${model_name_or_path} \
        --tokenizer_name ${model_name_or_path}

    cp ${output_path}/all_results.json  all_output_logs/${model_name_or_path}_${file_name}_$task@${seed}.json

done

done
