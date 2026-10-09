## Content
This folder contains the codes we use to try to augment a pre-trained DiT model with LSCs. Although our attempt has failed, we feel it might be helpful if we release this code as well. If you are interested, you can develop on this code and see if you can succeed.  

`train.py` is for the retrofitting. `train_new.py` is to train a model from scratch, but this requires huge training resources, so we have not successfully trained a model in such a way.  

## Implementation Details
`lsc_type`:

- basic: standard LSC.
- res: the zero-init residual LSC we propose in appendix.
- xxx-advanced: without `advanced`, dense and symmetric LSCs are used. With `advanced`, sparse LSCs are used. **NOTE:** we initially design this sparse LSC because we believed a similar design was used in U-Nets, as it's so common that when people depict a U-Net structure, there's only one LSC per resolution layer. However, we later found that U-Nets in fact use dense designs. You can check the following codes:

```python
# https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/unets/unet_2d_condition.py#L1201
res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
```

Note that the slicing uses `len(upsample_block.resnets)` instead of `1`? It means multiple skipped activations are grabbed here.
Then, in https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/unets/unet_2d_blocks.py#L2429:

```python3
res_hidden_states = res_hidden_states_tuple[-1]
res_hidden_states_tuple = res_hidden_states_tuple[:-1]

# ...

hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)  # this is a very typical LSC concatenation operation
```

The implementation in diffusers might be a bit misleading.
In their codes, first a BATCH of skipped activations are grabbed from the stack and all fed into a block (a.k.a., a resolution layer).
Then, in this block's own code, this batch is gradually broken down into individual activations and consumed by each block in this resolution layer.
If you only see the outer code, it's natural that you might think there's only one LSC per resolution layer, but that's not the fact.  

This also answers why there're two blocks per down-sampling resolution but three blocks per up-sampling resolution:
Each down-sampling block AND the downsampler outputs a skipped activation, so there need to be three up-sampling blocks to consume them.

In conclusion, DO NOT use `xxx-advanced` as it's an incorrect design.

`reg_type`:

- `ga`: gaussian_mse_loss
- `tv`: total_variation_loss
- `ss`: self_similarity_loss
- `wh`: whitening_loss_conv

`lsc_multiplier`: additional lr multiplier for LSC params.

## Usage
Prepare training data:
```bash
python3 generate_data.py --input_file datasets/diffusionDB/metadata.parquet --output_path datasets/lsc_training/pixart-sigma --slicing_count 4 --slicing_idx 0

# slicing_count and slicing_idx is used to run multiple processes in parallel
# e.g., you can set slicing_count=4 and start 4 processes with slicing_idx set to 0 - 3
```

Two-stage training as used in Skip-DiT:
```bash
# stage 1
python3 lsc/train.py --lr 5e-4 --accumulation_step 2 --input_root datasets/lsc_training/pixart-sigma --output_dir checkpoints/lsc_training/pixart-sigma --save_every_steps 1000

# stage 2
python3 lsc/train.py --lr 1e-5 --accumulation_step 1 --lsc_type basic-advanced --reg_type ga --reg_weight 0 --lsc_multiplier 10 --reg_layer vit-block10-out --start_from checkpoints/lsc_training/pixart-sigma/stage1/lr0.0005-basic-advanced-reg-vit-block14-out-['ga']-rw0.0-lm1.0/44000/lsc.safetensors --input_root datasets/lsc_training/pixart-sigma --output_dir checkpoints/lsc_training/pixart-sigma --save_every_steps 1000
```

Single-stage training as described in our appendix:
```bash
python3 lsc/train.py --lr 1e-5 --accumulation_step 1 --lsc_type res-advanced --reg_type ga --reg_weight -0.002 --reg_layer_list vit-block0-out vit-block2-out vit-block4-out vit-block6-out --reg_weight_2 0.002 --reg_layer_list_2 vit-block9-out vit-block10-out vit-block11-out vit-block12-out vit-block13-out vit-block14-out --lsc_multiplier 10 --input_root datasets/lsc_training/pixart-sigma --output_dir checkpoints/lsc_training/pixart-sigma --save_every_steps 1000

# resume from a checkpoint
python3 lsc/train.py --lr 1e-5 --accumulation_step 1 --lsc_type res --reg_type ga --reg_weight 0 --resume_from checkpoints/lsc_training/pixart-sigma/single-stage/res-reg-vit-block10-out-['ga']-rw0.001-lm2.0-rfNone/12000/ --reg_layer vit-block10-out --reg_layer_end vit-block16-out --lsc_multiplier 5 --input_root datasets/lsc_training/pixart-sigma --output_dir checkpoints/lsc_training/pixart-sigma --save_every_steps 1000
```

From-scratch training:
```bash
python3 lsc/train_new.py --lr 1e-5 --accumulation_step 1 --lsc_type basic --reg_type ga --reg_weight 0 --reg_layer vit-block10-out --reg_layer_end vit-block16-out --lsc_multiplier 1 --input_root lsc_training/pixart-sigma --output_dir checkpoints/lsc_training/pixart-sigma --save_every_steps 1000
```
