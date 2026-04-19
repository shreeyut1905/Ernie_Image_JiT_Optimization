# Ernie_Image_JiT_Optimization
for running jit
```bash
run.sh
```
for running seacache 
```bash
python ernie_seacache_generate.py --output_dir [save_dir] --prompt "a small boat on a calm lake at dawn" --height 512 --width 512 --num_inference_steps 50  --num_images_per_prompt 1 --guidance 4.0 --seacache_thresh 0.1
