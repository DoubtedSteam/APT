export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='8888'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='2'
python run.py with data_root=./arrows num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 \
load_path="results/finetune_irtr_f30k_randaug_seed0_from_vilt_200k_mlm_itm/version_5/checkpoints/epoch=9-step=5869.ckpt"