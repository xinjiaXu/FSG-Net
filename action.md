# FSG-Net Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/FSG-Net`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FSG-Net
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FSG-Net
git switch reproduce
git pull --ff-only
```

## Environment and pretrained backbone

```bash
bash setup_server_env.sh
conda activate fsgnet-rs
bash prepare_server_data.sh /home/xxj/Data
```

The model constructs timm ResNet-18 with `pretrained=True`. Environment setup warms the timm pretrained cache; it needs access to the weight source. No separate FSG-Net checkpoint is bundled.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

These commands use code defaults: ResNet-18, `fpn+aspp+fuse+drop`, FCN, BCE+Dice, batch 32, 10 epochs, input 256, lr `0.001`, weight decay `0.01`, and finetuning enabled. Training validates each epoch and saves the best-F1 checkpoint under `FSG-Net_work_dirs/<dataset>/train/<run_id>/weights/`. The server launcher uses one GPU by default; `GPU_IDS=0,1 bash run_train.sh levircd` exposes both to training.

## Evaluate

Pass the checkpoint created for each dataset:

```bash
bash run_eval.sh levircd FSG-Net_work_dirs/levircd/train/<run_id>/weights/<checkpoint>.pt
bash run_eval.sh levircdplus FSG-Net_work_dirs/levircdplus/train/<run_id>/weights/<checkpoint>.pt
bash run_eval.sh cdd FSG-Net_work_dirs/cdd/train/<run_id>/weights/<checkpoint>.pt
bash run_eval.sh whucd FSG-Net_work_dirs/whucd/train/<run_id>/weights/<checkpoint>.pt
bash run_eval.sh sysucd FSG-Net_work_dirs/sysucd/train/<run_id>/weights/<checkpoint>.pt
```

Evaluation is single-GPU and reports Precision, Recall, F1, mIoU, and OA. Results and prediction maps go under `FSG-Net_work_dirs/<dataset>/eval/<checkpoint-derived-name>/`.

## Parameters and FLOPs

After activating `fsgnet-rs`, install the profiler and count the default ResNet18 model with one paired 256x256 input:

```bash
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --device cuda:0 \
  --output FSG-Net_work_dirs/profile_256.json
```

The script uses the training defaults (ResNet18, `fpn+aspp+fuse+drop`, FCN, single label) and does not need a trained checkpoint. It reports `params_m` and `flops_g`; unsupported operations are listed, and FLOPs are partial if that list is non-empty. fvcore counts one multiply-add as one FLOP.
