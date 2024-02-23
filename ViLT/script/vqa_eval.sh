CUDA_VISIBLE_DEVICES=1 python run.py with data_root=./arrows num_gpus=1 num_nodes=1 per_gpu_batchsize=128 task_finetune_vqa_randaug test_only=True precision=32 \
load_path="lora_204/finetune_vqa_randaug_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/epoch=9-step=24989.ckpt"
