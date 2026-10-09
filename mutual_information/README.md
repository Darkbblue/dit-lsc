# Mutual Information Probing

## Run
```bash
# extract features
python3 generic-diffusion-feature/extract_feature_per_prompt.py --layer configs/config_dit_mi.json --version dit --img_size 256 --batch_size 1 --t 50 --input_dir "datasets/lsc_training/pixart-sigma/image/*.png" --prompt_file "datasets/lsc_training/pixart-sigma/prompt/*.txt" --output_dir outputs/mi/dit-wo-LSC/ --use_original_filename --sample_name_first

# MINE running
python3 generic-diffusion-feature/mutual_information/main.py --N 1000 --feat_root "outputs/mi/dit-wo-LSC/*" --layers configs/config_dit_mi.json --img_root datasets/lsc_training/pixart-sigma/image/
python3 generic-diffusion-feature/mutual_information/main.py --N 1000 --feat_root "outputs/mi/dit-wo-LSC/*" --layers configs/config_dit_mi.json --img_root datasets/lsc_training/pixart-sigma/image/ --critic_hidden 1024 --proj_dim 512 --train_steps 1000

```
