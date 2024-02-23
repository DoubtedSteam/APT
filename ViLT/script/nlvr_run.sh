export MASTER_ADDR='0.0.0.0'
# export MASTER_PORT='8000'
export MASTER_PORT='7003'
export NODE_RANK='0'
# export CUDA_VISIBLE_DEVICES='1' 
export CUDA_VISIBLE_DEVICES='3'   
python run.py with data_root=./arrows num_gpus=1 num_nodes=1 task_finetune_nlvr2_randaug per_gpu_batchsize=32 load_path="./vilt_200k_mlm_itm.ckpt" \
# learning_rate=1e-4

