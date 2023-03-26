# Kernel_raznos
### Skoltech ML course project, "tema_raznosa"

This is the page of our course project that is part of Skoltech's Machine Learning course. 
The main task is implement and benchmark different algorithms for "Image restoration" task i.e. deblurring.
In this repository you can check the results of our study.

### General project goals:

In this paper, the authors propose the model of non-uniform motion blur, which
generalize the  common used uniform model of blurring process. The key idea is to use not one kernel for the whole image, but per-pixel blur kernels. In this project you will deal with blind deblurring problem. Our team is supposed to code and reproduce the results of the authors for Kohler and Gopro datasets with real blur and compare them with DeblurGAN model.

To check our task click [here](https://docs.google.com/spreadsheets/d/1yvhUzqHK9bmbD7OdSE-DOcadRlaeC3xECUwOMFZgw-Q/edit#gid=0)

### References: 
1. Non-uniform Blur Kernel Estimation via Adaptive Basis Decomposition [arXiv Paper Version](https://arxiv.org/pdf/2102.01026.pdf)
2. DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks [arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)

### Project content: 
1. Our team's final presentation [link](Docs/ML2023_project18.pdf)
2. Scientific style final report [link](Docs/Final_report.pdf)
3. **Model trained from scratch by our team**




## Our results
<p align="center">
  <img width="300" src="Docs/Blurry1_1.png?raw=true">
  <br>
  <strong>Original Blurry Image</strong>
</p>

<p align="center">
  <img width="300" src="Docs/DGAN_deblur1.png?raw=true">
  <br>
  <strong>Deblurred Image using Deep Generative Adversarial Network (DeblurGAN)</strong>
</p>

<p align="center">
  <img width="300" src="Docs/Adaptive_deblur1.png?raw=true">
  <br>
  <strong>Deblurred Image using Adaptive Basis Estimation </strong>
</p>

## Our Metrics
<div align="center">
  
|                   | Blurry   | Hirsch   | DeblurGAN | Adaptive  |
| :---       | :--- | :--- | :--- | :--- | 
| **PSNR Expected**     | 27.58 | 33.16 | NA    | 35.19 | 
| **PSNR Calculated**   | 27.58 | 33.16 | 26.98 | 34.47 | 


</div>


<p align="center">
  <img width="500" src="Docs/train_pic.png?raw=true">
  <br>
  <strong> Our trained model </strong>
</p>

### Metric evaluation
Since the Kohler dataset is synthetic blurred with non uniform kernels, standart approaches to calculate PSNR does not show valid results, thus, we use MATLAB instruments to find PSNR and SSIM.
Here you can check instructions for benchmarking deblur algorithms on [Kohler dataset](https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake/#Image1_1)
## Tips

1. make sure that you have sufficient GPU with at least >4 gb of video memory, our setup had Nvidia gtx 1660(laptop) 6GB. Our recomendation is to use PC descrete GPU with ~12GB of videomemory.
3. The installation procces of CUDA and initiolizing of setup can be a little bit challenging, look for the Nvidia official [guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) how to install CUDA.
4. We reccomend you to use conda enviroment to setup CUDA+PyTorch setup, you can use PyTorch's [guide](https://pytorch.org/get-started/locally/) to do so. 

## Requirements
* scikit-image
* numpy
* torch==1.4.0
* torchvision==0.5.0
* opencv-python
* CUDA toolkit 11.6

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

Model can be downloaded from here ([dropbox](https://www.dropbox.com/s/ro9smg1i7lh5b8d/TwoHeads.pkl?dl=0))

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
* official KupynOrest git page [here](https://github.com/KupynOrest/DeblurGAN)
* useful fork with good readme and instructions [here](https://github.com/fatalfeel/DeblurGAN)
