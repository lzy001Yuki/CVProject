# Touch-and-Go

### [Dataset](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B) | [Website](https://touch-and-go.github.io/) |   [Paper](https://arxiv.org/pdf/2211.12498.pdf)
<br>

<img src='imgs/teaser.jpg' align="right" width=960>  
  

<br><br><br>
This repository contains the official PyTorch implementation of our applications paper [Touch and Go: Learning from Human-Collected Vision and Touch ](https://arxiv.org/pdf/2211.12498.pdf).


[Touch and Go: Learning from
Human-Collected Vision and Touch](https://arxiv.org/pdf/2211.12498.pdf)  
 [Fengyu Yang](https://fredfyyang.github.io/), [Chenyang Ma](https://www.linkedin.com/in/chenyang-ma-66945091), [Jiacheng Zhang](https://www.linkedin.com/in/jiacheng-zhang-689b8319a), [Jing Zhu](https://jwzhi.github.io/), [Wenzhen Yuan](http://robotouch.ri.cmu.edu/yuanwz/), [Andrew Owens](https://andrewowens.com/)<br>
University of Michigan and Carnegie Mellon University <br>
 In NeurIPS 2022 Datasets and Benchmarks Track


## Todo

- [x] Visuo-tacile Self-supervised Learning
- [x] Tactile-driven Image Stylization
- [x] Diffusion-based Tactile Encoder (training/eval scripts)

## Diffusion Tactile Encoder (new)

- Train the encoder with CLIP vision/text conditioning and InfoNCE alignment.
- Requirements: PyTorch, TorchVision, and OpenAI CLIP.

Install CLIP:

```
pip install git+https://github.com/openai/CLIP.git
```

Train:

```
python train.py \
  --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
  --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
  --split train \
  --use_text --text_from_label \
  --siglip_model google/siglip-base-patch16-224 \
  --t5_model t5-base \
  --use_cond_type_embed \
  --image_size 64 --batch_size 64 --epochs 20
```
```
torchrun --nproc_per_node=2 train.py \
  --distributed \
  --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
  --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
  --split train \
  --use_text --text_from_label \
  --siglip_model google/siglip-base-patch16-224 \
  --t5_model t5-base \
  --use_cond_type_embed \
  --image_size 64 --batch_size 64 --epochs 20 \
  --amp
```

```
uv run torchrun --nproc_per_node=3 train.py \
  --distributed \
  --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
  --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
  --split train \
  --siglip_model ./models/siglip-base-patch16-224 \
  --t5_model ./models/t5-base \
  --use_cond_type_embed \
  --image_size 224 --batch_size 128 --epochs 20 \
  --amp \
  --out_dir ckpt/train_dit_bs128_is224_nottext
```
export HF_HUB_OFFLINE=1
Evaluate (retrieval R@1 on test set):

```
python eval.py \
  --checkpoint /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/myTacEncoder/ckpt/train_dit_bs128_is224_nottext/stage1_epoch020.pt \
  --dataroot  /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
  --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
  --label full \
  --image_size 224 \
  --epochs 30 --batch_size 128 --lr 1e-2 --wd 0.0 --model "dit" --out_dir ckpt/dit_eval/dit_only_224_bs128_notext
```


uv run torchrun --nproc_per_node=2 train_dino.py \
  --distributed \
    --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
    --epochs 100 \
    --batch_size 64 \
    --image_size 64 \
    --device cuda \
    --amp \
    --distributed \
    --out_dir ckpt/dino_offline3 \
    --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset"




'''
uv run torchrun --nproc_per_node=3 multi_train.py  --distributed --tag_root /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset --octopi_root /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/octopi --encoder dit --image_size 224 --embed_dim 768 --epochs 100 --siglip_model ./models/siglip-base-patch16-224 --t5_model ./models/t5-base --batch_size 64 --lr 1e-4 --out_dir ckpt/checkpoints_multi --amp
'''



uv run torchrun --nproc_per_node=3 clip_dino.py \
  --distributed \
    --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
    --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
    --split train \
    --image_size 64 \
    --use_text --text_from_label \
    --siglip_model ./models/siglip-base-patch16-224 \
    --t5_model ./models/t5-base \
    --patch_size 8 \
    --embed_dim 768 \
    --local_blocks 2 \
    --transformer_layers 4 \
    --nhead 6 \
    --mlp_dim 1024 \
    --contrast_mode fused \
    --temperature 0.07 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --wd 0.05 \
    --pool mean \
    --out_dir ckpt/dino_clip_exp \
    --save_every 10 \
    --amp --load /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/myTacEncoder/ckpt/dino_offline3/dino_tactile_e100.pt --enable_dino

  python eval_dino.py \
    --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
    --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
    --label full \
    --checkpoint /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/myTacEncoder/ckpt/dino_clip_exp/dino_contrastive_epoch100.pt \
    --image_size 64 \
    --batch_size 256 \
    --epochs 50


uv run torchrun --nproc_per_node=4 clip_dit.py \
--distributed \
    --dataroot /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset \
    --list_dir "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset" \
    --split train \
    --image_size 224 \
    --use_text --text_from_label \
    --siglip_model ./models/siglip-base-patch16-224 \
    --t5_model ./models/t5-base \
    --embed_dim -1 \
    --depth 8 \
    --heads 8 \
    --timesteps 1000 \
    --pool mean \
    --stage1_ckpt /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/myTacEncoder/ckpt/offline_is224_bs128/stage1_epoch020.pt \
    --contrast_mode text \
    --temperature 0.07 \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-5 \
    --wd 0.05 \
    --out_dir ckpt/dit_clip_is224_bs128 \
    --save_every 10 \
    --amp

### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/2211.12498.pdf).
```
@inproceedings{
yang2022touch,
  title={Touch and Go: Learning from Human-Collected Vision and Touch},
  author={Fengyu Yang and Chenyang Ma and Jiacheng Zhang and Jing Zhu and Wenzhen Yuan and Andrew Owens},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

### Acknowledgments
We thank Xiaofeng Guo and Yufan Zhang for the extensive help with the GelSight sensor, and thank Daniel Geng, Yuexi Du and Zhaoying Pan for the helpful discussions. This work was supported in part by Cisco Systems and Wang Chu Chien-Wen Research Scholarship.
