# TactiLDM for VLA: Learning Unified Visual-Text-Tactile Representations to Empower Vision-Language-Action Policies

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

We use [Touch-and-Go](https://touch-and-go.github.io/) Dataset when training, so please first download it.

## Training

Our training process has two stages. Firstly train a tactile encoder and then integrate it into the Latent Diffusion Model framework.

### Stage 1

**Single GPU:**
```bash
python stage1.py \
    --dataroot /path/to/dataset \
    --list_dir "/path/to/dataset" \
    --siglip_model_path /path/to/siglip \
    --t5_model_path /path/to/t5 \
    --image_size 224 \
    --latent_dim 256 \
    --projection_dim 512 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir checkpoints/stage1
```

**Multi-GPU (Recommended):**
```bash
# Using torchrun directly
torchrun --nproc_per_node=4 --master_port 29502 stage1.py \
    --dataroot /path/to/dataset \
    --list_dir "/path/to/dataset" \
  --siglip_model /path/to/siglip \
  --t5_model /path/to/t5-base \
    --batch_size 64 \
    --epochs 200 --all --save_dir <save_dir>
```

### Stage 2

**Single GPU:**
```bash
python stage2.py \
    --dataroot /path/to/dataset \
    --list_dir "/path/to/dataset" \
    --autoencoder_checkpoint checkpoints/stage1/stage1_final.pt \
    --siglip_model_path /path/to/siglip \
    --t5_model_path /path/to/t5 \
    --image_size 224 \
    --latent_dim 256 \
    --base_channels 128 \
    --timesteps 1000 \
    --batch_size 32 \
    --epochs 200 \
    --lr 1e-4 \
    --save_dir <save_dir> \
    --autoencoder_checkpoint /path/to/stage1_checkpoint
```

**Multi-GPU (Recommended):**
```bash
# Using torchrun directly
torchrun --nproc_per_node=4 stage2.py \
    --dataroot /path/to/dataset \
    --list_dir "/path/to/dataset" \
  --siglip_model /path/to/siglip \
  --t5_model /path/to/t5-base\
    --autoencoder_checkpoint /path/to/stage1_checkpoint \
    --batch_size 32 \
    --epochs 200 --save_dir checkpoints/<save_dir>
```
## Evaluation （Linear Probing）


**Single GPU:**
```bash
python eval_linear_probe.py \
    --dataroot /path/to/dataset \
    --list_dir /path/to/lists \
    --ldm_checkpoint checkpoints/stage2/stage2_final.pt \
    --siglip_model_path /path/to/siglip \
    --t5_model_path /path/to/t5 \
    --label full \
    --batch_size 128 \
    --probe_epochs 30 \
    --probe_lr 1e-2 \
    --use_conditioning \
    --save_dir checkpoints/eval
```

**Multi-GPU (Recommended):**
```bash
# Using torchrun directly
torchrun --nproc_per_node=4 eval.py \
    --ldm_checkpoint /path/to/stage2_checkpoint \
    --dataroot /path/to/dataset \
    --list_dir "/path/to/dataset" \
  --siglip_model /path/to/siglip \
  --t5_model /path/to/t5-base \
    --label full \
    --batch_size 128 \
    --probe_epochs 30 \
    --use_conditioning --save_dir <save_dir>
```
