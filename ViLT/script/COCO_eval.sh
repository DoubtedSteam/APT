export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='8005'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='0'
python run.py with data_root=./arrows num_gpus=1 num_nodes=1 per_gpu_batchsize=8 task_finetune_irtr_coco_randaug test_only=True precision=32 \
load_path="result/finetune_irtr_coco_randaug_seed0_from_vilt_200k_mlm_itm/version_13/checkpoints/epoch=9-step=22139.ckpt"