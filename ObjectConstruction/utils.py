from math import log10, sqrt
from PIL import Image, ImageChops
from matplotlib import pyplot
import cv2
import cv2
import glob
import os
import numpy as np


class Utils:
    def __init__(self):
        self.original_imgpath = "./complete/original_image_00.jpg"
        self.generated_imgpath = "./complete/completed*.jpg"

    def run(self, original, generated):
        mse = np.mean((original - generated) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def psnr(self):
        psnr_list = []
        self.original = cv2.imread(self.original_imgpath)
        generated_list = glob.glob(self.generated_imgpath)
        for generated in generated_list:
            self.generated = cv2.imread(generated)
            value = self.run(self.original, self.generated)
            psnr_list.append(value)
        if psnr_list:
            max_psnr_value = max(psnr_list)
            print(max_psnr_value, psnr_list)
            return generated_list[psnr_list.index(max_psnr_value)]

    def image_sim(self):
        image_original = cv2.imread(self.original_imgpath)
        image_generated_list = glob.glob(self.generated_imgpath)
        percentage_list = []
        for idx in range(len(image_generated_list)):
            image_two = cv2.imread(image_generated_list[idx])
            res = cv2.absdiff(image_original, image_two)
            res = res.astype(np.uint8)
            percentage = round(100 - ((np.count_nonzero(res) * 100) / res.size), 2)
            percentage_list.append(percentage)
        max_percentage = np.max(percentage_list)
        print(f'Maximum similarity percentage is {max_percentage}\nPercentage list: {percentage_list}\nImage path: {image_generated_list[percentage_list.index(max_percentage)]}')
        return image_generated_list[percentage_list.index(max_percentage)]

    def image_diff(self):
        image_one = Image.open(self.original_imgpath)
        image_two = Image.open(self.generated_imgpath)

        diff = ImageChops.difference(image_one, image_two)
        if diff.getbbox():
            diff.save("./images/diff.png")

    def plot_history(self, d_hist, g_hist):
        pyplot.subplot(1, 1, 1)
        pyplot.plot(d_hist, label='discriminator loss')
        pyplot.plot(g_hist, label='generator loss')
        pyplot.legend()
        # save plot to file
        pyplot.savefig('images/plot_line_plot_loss.png')
        pyplot.close()


# Utils().image_sim()
# Utils().image_diff()
