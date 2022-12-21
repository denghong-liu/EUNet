# EUNet

Pytorch implementation of EUNet

## Requirements
- Python 3.9
- [Pytorch](https://pytorch.org/) 1.12.1

## Training
To train EUNet, run the following commands. You may need to change the `dir_data`, `dataset_name`, `scale`, `n_colors`, `is_blur`, `learning_rate`, etc. in the option.py file for different settings. 

```python
# Bicubic downsampling
python main.py --scale 2 --dir_data hdata/data/ --dataset_name Pavia  --n_colors 102

# Gaussian downsampling
python main.py --scale 2 --dir_data hdata/data/ --dataset_name Pavia  --n_colors 102 --is_blur True --learning_rate 1e-3
```

## Testing
For your convience, we provide the testset of Pavia Centre in `/hdata/data/` and the pretrained 2X model in `/hsr/model/`.

```python
python main_test.py --scale 2 --dir_data hdata/data/ --dataset_name Pavia  --n_colors 102 --model_path hsr/model/G.pth
```

## References
- [KAIR](https://github.com/cszn/KAIR)
