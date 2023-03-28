import json
import re
import pickle as pkl
import shutil

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

DATA_PATH = "ADE20K_2021_17_01"
INDEX_FILE = "index_ade20k.pkl"
FILTERD_DATA_PATH = "my_data/"

def filter_dataset():

    with open(f"{DATA_PATH}/{INDEX_FILE}", "rb") as f:
        index_ade20k = pkl.load(f)

    images_ids = []

    for i in range(len(index_ade20k["filename"])):
        full_file_name = f"{index_ade20k['folder'][i]}/{index_ade20k['filename'][i]}"
        json_name = full_file_name.replace(".jpg", ".json")

        with open(json_name, encoding="utf-8") as json_file:
            img_desc = json.load(json_file)
            for seg_object in img_desc["annotation"]["object"]:
                if seg_object["name_ndx"] in [401, 1831]: # car or person
                    mask_filename = re.sub("ADE_.*", seg_object["instance_mask"], json_name)
                    mask = cv2.imread(mask_filename)
                    object_size = np.sum(mask == 255)
                    if object_size > 400:
                        images_ids.append(full_file_name)
                        shutil.copyfile(full_file_name, f"my_data/{index_ade20k['filename'][i]}")

def object_convolution(image, kernel, mask_object):
    iH, iW, iCh = mask_object.shape
    _, kW = kernel.shape
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW, iCh)) 

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            if (mask_object[y - pad][x - pad] != 0).all():
                for ch in range(iCh):
                    roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1, ch]
                    k = (roi * kernel).sum()
                    output[y - pad, x - pad, ch] = k
            else:
                output[y - pad, x - pad] = image[y, x]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

def get_trajectory(canvas=64, iters=2000, max_len=60, expl=None):
    if expl is None:
            expl = 0.005
    
    tot_length = 0
    big_expl_count = 0
    centripetal = 0.7 * np.random.uniform(0, 1)
    prob_big_shake = 0.2 * np.random.uniform(0, 1)
    gaussian_shake = 10 * np.random.uniform(0, 1)
    init_angle = 360 * np.random.uniform(0, 1)

    img_v0 = np.sin(np.deg2rad(init_angle))
    real_v0 = np.cos(np.deg2rad(init_angle))

    v0 = complex(real=real_v0, imag=img_v0)
    v = v0 * max_len / (iters - 1)

    if expl > 0:
        v = v0 * expl

    x = np.array([complex(real=0, imag=0)] * (iters))

    for t in range(0, iters - 1):
        if np.random.uniform() < prob_big_shake * expl:
            next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
            big_expl_count += 1
        else:
            next_direction = 0

        dv = next_direction + expl * (
            gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
            max_len / (iters - 1))

        v += dv
        v = (v / float(np.abs(v))) * (max_len / float((iters - 1)))
        x[t + 1] = x[t] + v
        tot_length = tot_length + abs(x[t + 1] - x[t])

    x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
    x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
    x += complex(real=np.ceil((canvas - max(x.real)) / 2), imag=np.ceil((canvas - max(x.imag)) / 2))

    return x
    

def get_kernel(canvas, trajectory=None, fraction=None):
    if trajectory is None:
        trajectory = get_trajectory(canvas, expl=0.005)
    if fraction is None:
        fraction = [1/100, 1/10, 1/2, 1]


    iters = len(trajectory)
    PSFs = []
    canvas = (canvas, canvas)
    psf = np.zeros(canvas)

    triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
    triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
    for j in range(len(fraction)):
        if j == 0:
            prevT = 0
        else:
            prevT = fraction[j - 1]

        for t in range(len(trajectory)):
            if (fraction[j] * iters >= t) and (prevT * iters < t - 1):
                t_proportion = 1
            elif (fraction[j] * iters >= t - 1) and (prevT * iters < t - 1):
                    t_proportion = fraction[j] * iters - (t - 1)
            elif (fraction[j] * iters >= t) and (prevT * iters < t):
                t_proportion = t - (prevT * iters)
            elif (fraction[j] * iters >= t - 1) and (prevT * iters < t):
                t_proportion = (fraction[j] - prevT) * iters
            else:
                t_proportion = 0

            m2 = int(np.minimum(canvas[1] - 1, np.maximum(1, np.math.floor(trajectory[t].real))))
            M2 = int(m2 + 1)
            m1 = int(np.minimum(canvas[0] - 1, np.maximum(1, np.math.floor(trajectory[t].imag))))
            M1 = int(m1 + 1)

            psf[m1, m2] += t_proportion * triangle_fun_prod(trajectory[t].real - m2, trajectory[t].imag - m1)
            psf[m1, M2] += t_proportion * triangle_fun_prod(trajectory[t].real - M2, trajectory[t].imag - m1)
            psf[M1, m2] += t_proportion * triangle_fun_prod(trajectory[t].real - m2, trajectory[t].imag - M1)
            psf[M1, M2] += t_proportion * triangle_fun_prod(trajectory[t].real - M2, trajectory[t].imag - M1)

        PSFs.append(psf / (iters))
    
    return PSFs

def filter_dataset():

    with open(f"{DATA_PATH}/{INDEX_FILE}", "rb") as f:
        index_ade20k = pkl.load(f)

    num_of_images = 0

    for i in range(len(index_ade20k["filename"])):
        full_file_name = f"{index_ade20k['folder'][i]}/{index_ade20k['filename'][i]}"
        json_name = full_file_name.replace(".jpg", ".json")

        with open(json_name, encoding="utf-8") as json_file:
            img_desc = json.load(json_file)
            for seg_object in img_desc["annotation"]["object"]:
                if seg_object["name_ndx"] in [401, 1831]: # car or person
                    mask_filename = re.sub("ADE_.*", seg_object["instance_mask"], json_name)
                    mask = cv2.imread(mask_filename)
                    object_size = np.sum(mask == 255)
                    if object_size > 400:
                        num_of_images += 1
                        shutil.copyfile(full_file_name, f"{FILTERD_DATA_PATH}/{index_ade20k['filename'][i]}")
    return num_of_images



def main():
    num = filter_dataset()
    print(f"Choosed {num} images from dataset")



if __name__ == "__main__":
    main()
