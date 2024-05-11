
GPU_NUM=1
TOTAL_GPU=1

checkpoint_dir='mPLUG2_MSRVTT_QA.pth'
output_dir='output/videoqa_msrvtt_'${TOTAL_GPU}

python -u -m torch.distributed.launch --nproc_per_node=1 \
	--nnodes=1 \
	--node_rank=1 \
    --use_env \
    test_loadmodel.py \
    --config configs_video/VideoQA_msrvtt_large.yaml \
    --text_encoder bert-large-uncased \
    --text_decoder bert-large-uncased \
    --output_dir ${output_dir} \
    --checkpoint ${checkpoint_dir} \
    --do_two_optim \
    --evaluate