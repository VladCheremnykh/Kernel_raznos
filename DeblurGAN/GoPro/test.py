import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from ssim import SSIM
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

# for i, data in enumerate(dataset):
for i, data in enumerate(dataset, 1):

	if i >= opt.how_many:
		break
	counter = i
	model.set_input(data)
	model.test()
	visuals = model.get_current_visuals()

	avgPSNR += PSNR(visuals['fake_B'],visuals['real_A'])
	pilFake = Image.fromarray(visuals['fake_B'])
	pilReal = Image.fromarray(visuals['real_A'])
	avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
	img_path = model.get_image_paths()
	print('process image... %s' % img_path)
	visualizer.save_images(webpage, visuals, img_path)
	
avgPSNR /= counter
avgSSIM /= counter

metric_path = 'C:/Users/Admin/Documents/Skoltech/ML/Project/DeblurGAN-master/results/GoPro_sharp_metrics.txt'
with open(metric_path, 'w') as f:
	f.write(f"PSNR = {avgPSNR}\n")
	f.write(f"SSIM = {avgSSIM}\n")
	
print('PSNR = %f, SSIM = %f' %
				  (avgPSNR, avgSSIM))

webpage.save()
