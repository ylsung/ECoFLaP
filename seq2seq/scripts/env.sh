declare -A num_epochs

num_epochs["rte"]=20
num_epochs["qnli"]=10
num_epochs["mnli"]=10
num_epochs["cola"]=20
num_epochs["sst2"]=10
num_epochs["mrpc"]=20
num_epochs["qqp"]=10
num_epochs["stsb"]=20
num_epochs["superglue-boolq"]=20
num_epochs["superglue-multirc"]=10
num_epochs["superglue-wic"]=20
num_epochs["superglue-cb"]=20
num_epochs["superglue-copa"]=20
num_epochs["superglue-record"]=10
num_epochs["squad_v2"]=2
num_epochs["winogrande_debiased"]=20

declare -A max_source_length
max_source_length["rte"]=128
max_source_length["qnli"]=128
max_source_length["mnli"]=128
max_source_length["cola"]=128
max_source_length["sst2"]=128
max_source_length["mrpc"]=128
max_source_length["qqp"]=128
max_source_length["stsb"]=128
max_source_length["superglue-boolq"]=128
max_source_length["superglue-multirc"]=128
max_source_length["superglue-wic"]=128
max_source_length["superglue-cb"]=128
max_source_length["superglue-copa"]=128
max_source_length["superglue-record"]=128
max_source_length["squad_v2"]=384
max_source_length["winogrande_debiased"]=128

declare -A batch_size
batch_size["rte"]=100
batch_size["qnli"]=100
batch_size["mnli"]=100
batch_size["cola"]=100
batch_size["sst2"]=100
batch_size["mrpc"]=100
batch_size["qqp"]=100
batch_size["stsb"]=100
batch_size["superglue-boolq"]=100
batch_size["superglue-multirc"]=100
batch_size["superglue-wic"]=100
batch_size["superglue-cb"]=100
batch_size["superglue-copa"]=100
batch_size["superglue-record"]=100
batch_size["squad_v2"]=32
batch_size["winogrande_debiased"]=100

declare -A learning_rate
learning_rate["rte"]=3e-4
learning_rate["qnli"]=3e-4
learning_rate["mnli"]=3e-4
learning_rate["cola"]=3e-4
learning_rate["sst2"]=3e-4
learning_rate["mrpc"]=3e-4
learning_rate["qqp"]=3e-4
learning_rate["stsb"]=3e-4
learning_rate["superglue-boolq"]=3e-4
learning_rate["superglue-multirc"]=3e-4
learning_rate["superglue-wic"]=3e-4
learning_rate["superglue-cb"]=1e-3
learning_rate["superglue-copa"]=3e-4
learning_rate["superglue-record"]=3e-4
learning_rate["squad_v2"]=3e-4
learning_rate["winogrande_debiased"]=3e-4

# declare -A wwwwww
# wwwwww["rte"]=zzz
# wwwwww["qnli"]=zzz
# wwwwww["mnli"]=zzz
# wwwwww["cola"]=zzz
# wwwwww["sst2"]=zzz
# wwwwww["mrpc"]=zzz
# wwwwww["qqp"]=zzz
# wwwwww["stsb"]=zzz
# wwwwww["superglue-boolq"]=zzz
# wwwwww["superglue-multirc"]=zzz
# wwwwww["superglue-wic"]=zzz
# wwwwww["superglue-cb"]=zzz
# wwwwww["superglue-copa"]=zzz
# wwwwww["superglue-record"]=zzz
# wwwwww["squad_v2"]=zzz
# wwwwww["winogrande_debiased"]=zzz
