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

file_name=lst

lst_reduction_factor=8

for seed in 0
do

for task in "qnli" "sst2" "mnli" "qqp"

do

    rm -rf outputs/${file_name}/
    
    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py \
        --training_config_file configs/${file_name}.json \
        --task_name $task --eval_dataset_name $task --test_dataset_name $task \
        --seed ${seed}  \
        --num_train_epochs ${num_epochs[$task]} \
        --lst_reduction_factor ${lst_reduction_factor} \
        --output_dir outputs/${file_name} \
        --gate_type "all_from_backbone"

    cp outputs/${file_name}/all_results.json  all_output_logs/linear_$task@${seed}.json

done
done