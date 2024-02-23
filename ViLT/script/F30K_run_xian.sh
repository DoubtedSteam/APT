cd /home/shamdxxxxyzlstdui/qiong/ViLT
conda activate vilt
python run.py with data_root=./arrows num_gpus=4 num_nodes=1 task_finetune_irtr_f30k_randaug per_gpu_batchsize=4 load_path="./vilt_200k_mlm_itm.ckpt"