# Kernel_raznos
### Skoltech ML course project, "tema_raznosa"

This is the course project page of Skoltech's Machine Learning course. 
The main task is implement and benchmark different algorithms for "Image restoration" task i.e. deblurring
In this repository you can check the results of our study.
Report.pdf depicts scientific format of our report, while "presentation.pdf" shows our team presentation


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
  

    
