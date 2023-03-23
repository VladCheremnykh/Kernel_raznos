# Kernel_raznos
### Skoltech ML course project, "tema_raznosa"

This is the course project page of Skoltech's Machine Learning course. 
The main task is implement and benchmark different algorithms for "Image restoration" task i.e. deblurring
In this repository you can check the results of our study.
Report.pdf depicts scientific format of our report, while "presentation.pdf" shows our team presentation

To check our task click [here](https://docs.google.com/spreadsheets/d/1yvhUzqHK9bmbD7OdSE-DOcadRlaeC3xECUwOMFZgw-Q/edit#gid=0)

References: 
1. Non-uniform Blur Kernel Estimation via Adaptive Basis Decomposition [arXiv Paper Version](https://arxiv.org/pdf/2102.01026.pdf)
2. DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks [arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)




## Non-uniform Blur Kernel Estimation via Adaptive Basis Decomposition



## Network Architecture

<p align="center">
<img width="900" src="Docs/architecture.png?raw=true">
  </p>
  
## Getting started



### Clone Repository
```
git clone https://github.com/GuillermoCarbajal/NonUniformBlurKernelEstimationViaAdaptiveBasisDecomposition
```

### Download the pretrained model

Model can be downloaded from [here](https://www.dropbox.com/s/ro9smg1i7lh5b8d/TwoHeads.pkl?dl=0)
### Compute kernels from an image
```
python compute_kernels.py -i image_path -m model_path
```


### Deblur an image or a list of images
```
python image_deblurring.py -b blurry_img_path --reblur_model model_path --output_folder results
```

### Parameters
Additional options:   
  `--blurry_images`: may be a singe image path or a .txt with a list of images.
  
  `--n_iters`: number of iterations in the RL optimization (default 30)       
  
  `--resize_factor`: input image resize factor (default 1)     
  
  `--saturation_method`: `'combined'` or `'basic'`. When `'combined'` is passed RL in the presence of saturated pixels is applied. Otherwise,  simple RL update rule is applied in each iteration. For Kohler images, `'basic'` is applied. For RealBlur images `'combined'` is better.
  
  `--gamma_factor`: gamma correction factor. By default is assummed `gamma_factor=2.2`. For Kohler dataset images `gamma_factor=1.0`.
  

# DeblurGAN
[arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)

Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.

Our network takes blurry image as an input and procude the corresponding sharp estimate, as in the example:
<img src="images/animation3.gif" width="400px"/> <img src="images/animation4.gif" width="400px"/>

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

### Prerequisites data
- cd ~/DeblurGAN
- ./install_data.sh
- or
- bash -x ./install_data.sh

### Train
- step 1 `open terminal`
- step 2 `pip3 install visdom`
- step 3 `python3 -m visdom.server`
- step 4 `open another terminal`
- step 5 `cd ~/DeblurGAN`
- step 6 `python3 ./train.py --dataroot ./data/combined --resize_or_crop crop --cuda True`
- If you do not want to use visdom.server then skip step 1~6 and use these commands
- `python3 ./train.py --dataroot ./data/combined --resize_or_crop crop --display_id -1 --cuda True`
- [----------Resume training--------------]
- `python3 ./train.py --dataroot ./data/combined --resize_or_crop crop --display_id -1 --cuda True --resume True`
- [----------FPN101 and Wgan-gp------]
- `python3 ./train.py --dataroot ./data/combined --resize_or_crop crop --display_id -1 --cuda True --which_model_netG FPN101 --gan_type wgan-gp`

### Test
- `python3 ./test.py --dataroot ./data/blurred --model test --dataset_mode single --cuda True`
- [----------FPN101----------]
- `python3 ./test.py --dataroot ./data/blurred --model test --dataset_mode single --cuda True --which_model_netG FPN101`

### Model trained 2000 times
https://drive.google.com/file/d/1vGiqFXa177sCGHEuKhDKQ0VxvZZ2qpZg

### **Model trained from scratch by our team**

"here"

### Metric evaluation
Since the Kohler dataset is synthetic blurred with non uniform kernels, standart approaches to calculate PSNR does not show valid results, thus, we use MATLAB instruments to find PSNR and SSIM.
Here you can check instructions for benchmarking deblur algorithms on [Kohler dataset](https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake/#Image1_1)
