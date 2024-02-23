export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='8005'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='2,3'
python run.py with data_root=./arrows num_gpus=2 num_nodes=1 task_finetune_irtr_coco_randaug per_gpu_batchsize=4 load_path="./vilt_200k_mlm_itm.ckpt"