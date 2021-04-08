

# Results

## Classification

CIFAR100

```bash
# SGD
python train.py --model vgg16 --epochs 200 --lr 0.05 --wd 5e-4  --output-dir vgg16_swa_e300 --device cuda:1

# SWA
python train.py --model vgg16 --epochs 300 --lr 0.05 --wd 5e-4  --output-dir vgg16_swa_e300 --device cuda:1 --swa --swa-lr 0.01 --swa-start 160
```

|     | VGG-16     |
| --- | ---------- |
| SGD | 71.9       |
| SWA | 74.5       |


## Segmentation


