psnr_sum = 0.0
ssim_sum = 0.0
count = 0

# Read file and calculate mean values
with open('output_metrics.txt', 'r') as f:
    for line in f:
        if line.startswith('PSNR:'):
            psnr_sum += float(line.split(':')[1])
        elif line.startswith('SSIM:'):
            ssim_sum += float(line.split(':')[1])
            count += 1

mean_psnr = psnr_sum / count
mean_ssim = ssim_sum / count

# Write mean values to the file
with open('output_metrics.txt', 'a') as f:
    f.write('\nMean PSNR: {}\n'.format(mean_psnr))
    f.write('Mean SSIM: {}\n'.format(mean_ssim))
