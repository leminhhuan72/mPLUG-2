python -u -m torch.distributed.launch --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
	--master_port=8888 \
	--nnodes=1 \
	--node_rank=1 \
    --use_env \
    test_dataloader.py \
    --config configs_video/VideoQA_msrvtt_large.yaml \
    --text_encoder bert-large-uncased \
    --text_decoder bert-large-uncased \