#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_retrieval_flickr.py --p 0.5 --epoch 12 \
--pretrained pretrained/model_base_retrieval_flickr.pth --config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr_compression_2x