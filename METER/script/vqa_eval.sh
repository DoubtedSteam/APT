export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='8000'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='2,3'
python run.py with data_root=./arrows num_gpus=2 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=64 clip16 text_roberta image_size=384 test_only=True \
load_path='lora_27/finetune_vqa_randaug_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/epoch=9-step=24989.ckpt' \
