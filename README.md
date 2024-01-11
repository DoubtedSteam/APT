> **Approximated Prompt Tuning for Vision-Language Pre-trained Models**
> [Qiong Wu](https://scholar.google.com/citations?hl=en&user=HyKLYKYAAAAJ)<sup>12</sup>,  Yiyi Zhou<sup>12</sup>, Shubin Huang<sup>1</sup>, Pingyang Dai<sup>1</sup>, [Rongrong Ji](https://mac.xmu.edu.cn/rrji/)<sup>12</sup>
><sup>1</sup>Media Analytics and Computing Lab, Department of Artificial Intelligence, School of Informatics, Xiamen University  
> <sup>2</sup>Institute of Artificial Intelligence, Xiamen University 

Prompt tuning is a parameter-efficient way to deploy large-scale pre-trained models to downstream tasks by adding task-specific tokens. In terms of vision-language pre-trained (VLP) models, prompt tuning often requires a large number of learnable tokens to bridge the gap between the pre-training and downstream tasks, which greatly exacerbates the already high computational overhead. In this paper, we revisit the principle of prompt tuning for Transformer-based VLP models, and reveal that the impact of soft prompt tokens can be actually approximated via independent information diffusion steps. In this case, we can avoid the expensive global attention modeling and reduce the computational complexity to a large extent. Based on this observation, we propose a novel Approximated Prompt Tuning (APT) approach towards efficient VL transfer learning. To validate APT, we apply it to a set of representative VLP models, and conduct extensive experiments on a bunch of downstream tasks. Meanwhile, the generalization of APT is also validated on CLIP for image classification and StableDiffusion for subject-driven text-to-image generation. The experimental results not only show the superior performance and computation efficiency of APT against the conventional prompt tuning methods, *e.g.*, +7.01% accuracy and -82.30% computation overhead on VQA, but also confirm its merits over other parameter-efficient transfer learning approaches.

---

## Setup
### Install

```bash
cd METER
pip install -r requirements.txt
pip install -e .
```

### Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Fine-tuning on Downstream Tasks

Work on the METER:

```bash
cd METER
```

Work on the ViLT:

```bash
cd ViLT
```

### VQAv2

#### Train

```bash
sh script/vqa_run.sh
```

#### Evaluate

Add the path of checkpoint to vqa_eval.sh.

```bash
sh script/vqa_eval.sh
```

### Flickr30k IR/TR

#### Train

```bash
sh script/F30K_run.sh
```

#### Evaluate

Add the path of checkpoint to F30K_eval.sh.

```bash
sh script/F30K_eval.sh
```

### NLVR2

#### Train

```bash
sh script/nlvr_run.sh
```

#### Evaluate

Add the path of checkpoint to nlvr_eval.sh.

```bash
sh script/nlvr_eval.sh
```

### Experiment Result

#### ViLT

| **Method**         | **Updated Parameter** | **Additional FLOPs** | **VQA test-dev** | **NLVR$^2$ test-P** | **Flickr30K IR R@1** | **Flickr30K TR R@1** | **Avg.** |
|--------------------|------------------------|-----------------------|------------------|---------------------|-----------------------|-----------------------|---------|
| Full Tuning        | 115.43M                | 0.0                   | 71.26            | 76.13               | 64.40                 | 83.50                 | 73.82   |
| Classifier Only    | -                      | 0.0                   | 65.75            | 66.08               | 57.42                 | 78.00                 | 66.81   |
| Deep Prompt        | 1.84M                  | 5.14G                 | 69.30            | 73.34               | 58.64                 | 79.50                 | 70.20   |
| LoRA               | 0.15M                  | 0.0                   | 68.44            | 72.77               | 57.44                 | 77.70                 | 69.09   |
| Scaled PA          | 1.80M                  | 0.44G                 | 70.40            | 75.13               | 61.88                 | 79.00                 | 71.60   |
| Adapter            | 3.56M                  | 0.86G                 | 70.85            | 75.51               | 62.68                 | 81.40                 | 72.61   |
| **APT**            | 1.92M                  | 0.91G                 | **70.94**        | **75.92**           | **63.26**             | **81.60**             | **72.93**|

#### METER

| **Method**         | **Updated Parameter** | **Additional FLOPs** | **VQA test-dev** | **NLVR$^2$ test-P** | **Flickr30K IR R@1** | **Flickr30K TR R@1** | **Avg.** |
|--------------------|------------------------|-----------------------|------------------|---------------------|-----------------------|-----------------------|---------|
| Full Tuning        | 323.31M                | 0.0                   | 77.43            | 83.05               | 82.22                 | 94.30                 | 84.25   |
| Classifier Only    | -                      | 0.0                   | 69.93            | 73.23               | 78.80                 | 89.00                 | 77.74   |
| Deep Prompt        | 3.68M                  | 13.05G                | 67.57            | 65.79               | 70.90                 | 87.70                 | 72.99   |
| LoRA               | 0.29M                  | 0.0                   | 74.00            | 78.82               | 79.86                 | 92.60                 | 81.32   |
| Adapter            | 5.34M                  | 1.64G                 | 74.70            | 79.93               | 80.38                 | 91.90                 | 81.73   |
| Scaled PA          | 3.82M                  | 1.12G                 | **75.36**        | **79.86**           | **80.30**             | **91.80**             | **81.83**|
| **APT**            | 3.83M                  | 2.31G                 | **75.45**        | **80.97**           | **80.88**             | **92.90**             | **82.55**|


## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE) and [METER](https://github.com/zdou0830/METER/tree/main) licensed under [MIT](https://github.com/zdou0830/METER/blob/main/LICENSE) and some of the code is borrowed from [CLIP](https://github.com/openai/CLIP) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).