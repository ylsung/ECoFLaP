#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_nlvr.py --p 0.9 --epoch 27 \
--pretrained pretrained/model_base_nlvr.pth --config ./configs/nlvr.yaml \
--output_dir output/nlvr_nlvr2_compression_10x
