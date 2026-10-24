# FSG-Net server setup

FSG-Net reads image names from each split's image-A directory and pairs them with same-named files in image-B and the label directory. Although the supplied datasets also have split text files, this loader does not consume them. The adapter reads the existing folders directly; it creates no copies or compatibility trees.

## Environment

From `/home/xxj/code/DINOCLIP_comparative_test/FSG-Net`:

```bash
bash setup_server_env.sh
conda activate fsgnet-rs
```

The setup uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8 for the RTX 5090. It keeps the repository's `timm==0.6.13` model API and installs the other pinned dependencies through `requirements_server.txt`. FSG-Net creates a timm `resnet18` with `pretrained=True`; the setup script warms the timm pretrained-weight cache. That step needs access to the timm weight source. Once cached, the training script reuses those same ImageNet weights.

## Data layout

Verify the expected paths:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

| Key | Dataset root | Image A | Image B | Label |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

Labels are binarized by mapping non-zero values to change. CDD's `OUT` folder is passed as the label directory.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers preserve `train.py` defaults: ResNet-18, `fpn+aspp+fuse+drop` neck, FCN head, BCE+Dice loss, batch size 32, 10 epochs, input size 256, learning rate `0.001`, weight decay `0.01`, and finetuning enabled. The code accumulates gradients toward a total batch of 32. It evaluates the validation split each epoch and saves the best-F1 checkpoint. Runtime worker count defaults to 8, and the visible GPU defaults to 0; set `GPU_IDS=0,1` to use both GPUs.

The upstream model selects CUDA on every visible GPU when more than one is available. The server adapter changes the previous four-device hardcoding to use the number of visible GPUs, so one or two GPUs can run.

Outputs are placed under `FSG-Net_work_dirs/<dataset>/train/<run_id>/`, including weights and `epoch_process.csv`.

## Evaluate

```bash
bash run_eval.sh levircd FSG-Net_work_dirs/levircd/train/1/weights/epoch_x_x.pt
```

Replace the checkpoint path with an actual saved `.pt` file. Evaluation reports Precision, Recall, F1, mIoU, and OA, writes `eval_metrics.csv`, and saves prediction maps under the checkpoint's evaluation directory. Evaluation uses one GPU; select it with `GPU_IDS=1` if needed.
