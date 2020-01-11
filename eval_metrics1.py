# import the necessary packages
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
from numba import autojit, prange
#from skimage.measure import structural_similarity as ssim

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
    ms = mse(imageA, imageB)
    ssim = measure.compare_ssim(imageA, imageB)
    psnr=cv2.PSNR(imageA,imageB)
    
    
    print(title)
#    s = ssim(imageA, imageB)
#    print("PSNR = \n", ps, "\nMSE = \n", m, "\nSSIM = \n", s)
    return (ms, ssim, psnr)
	# setup the figure
#    fig = plt.figure(title)
#    plt.suptitle("MSE: %.2f, PSNR: %.2f" % (m, ps))

	# show first image
#    ax = fig.add_subplot(1, 2, 1)

#    plt.imshow(imageA, cmap = plt.cm.gray)
#    plt.axis("off")

	# show the second image
#    ax = fig.add_subplot(1, 2, 2)
#    plt.imshow(imageB, cmap = plt.cm.gray)
#    plt.axis("off")

	# show the images
#    plt.show()
def main():
    
    orig_dir = sys.argv[1]
    fake_dir = sys.argv[2]
    
    print("Please 'DON'T' give '/' at the end of directory path")
    print("Also, pls. ensure that the image_names of real and synthesized are 'exactly' the same")

    path_orig, root_orig, files_orig = next(os.walk(orig_dir))
#    path_fake, root_fake, files_fake = next(os.walk(fake_dir))
    n = len(files_orig)

    print(n)
    orig_sum_mse = 0
    orig_sum_ssim = 0
    orig_sum_psnr = 0
    fake_sum_mse = 0
    fake_sum_ssim = 0
    fake_sum_psnr = 0
    for i in prange(n):
        img_name = files_orig[i]

        img_path_orig =  orig_dir + '/' + img_name
        img_path_fake =  fake_dir + '/' + img_name

        original=cv2.imread(img_path_orig,0)
        contrast=cv2.imread(img_path_fake,0)
	# # convert the images to grayscale
	# #original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	# #contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

	# compare the images
        print("For image ", img_path_orig)    
        orig_results = compare_images(original, original, "Original vs. Original")
        orig_sum_mse += orig_results[0]
        orig_sum_ssim += orig_results[1]
        orig_sum_psnr += orig_results[2]

        fake_results = compare_images(original, contrast, "Original vs. Generated")
        fake_sum_mse += fake_results[0]
        fake_sum_ssim += fake_results[1]
        fake_sum_psnr += fake_results[2]
	#compare_images(original, shopped, "Original vs. Generated")

    print("Original vs Original results")
    print("mse = \n", orig_sum_mse/n, "\npsnr = \n", orig_sum_psnr/n, "\nssim = \n", orig_sum_ssim/n)
    print("Original vs Generated results")
    print("mse = \n", fake_sum_mse/n, "\npsnr = \n", fake_sum_psnr/n, "\nssim = \n", fake_sum_ssim/n)

if __name__=="__main__":
#	original=cv2.imread("/Users/atishpatel/Downloads/real.jpg",0)
#	contrast=cv2.imread(,0)
    main()

