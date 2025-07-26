# GTSD

This repo provides the code and models for [GTSD: Generative Text Steganography Based on Diffusion Model](https://arxiv.org/abs/2504.19433). 

## ⚙️ Experiment Preparation

## Training

Start train.sh. We use *Movie* dataset as an example to demonstrate the process of GTSD.

``` 
bash train.sh
```

## Gneration

Strat gnoise.sh. We use the pre-trained diffusion model to generate stego texts of *Movie* dataset. The prompts of stego texts depends on secret bits.

Please 

``` 
bash gnoise.sh
```

The path of stego texts in **./my_output/movie/commongen/120000_ema_0.9999_skip__xy_20/num10**

The path of prompt in **./data/movie**


