
import random
import numpy as np

for case in range(1,7):
    for T in range(0,10):
        img = np.load("E:/dir_lab_pre/creatis/case%d/case%d_T%d0.npy"%(case,case,T))
        noise_img = img
        height, width, deep = noise_img.shape[0], noise_img.shape[1], noise_img.shape[2]
        num = int(height * width * deep * 0.05)  # 多少个像素点添加椒盐噪声
        for i in range(num):
            w = random.randint(0, width - 1)
            h = random.randint(0, height - 1)
            d = random.randint(0, deep - 1)
            if random.randint(0, 1) == 0:
                noise_img[h, w, d] = 0
            else:
                noise_img[h, w, d] = 255
        np.save("E:/dir_lab_pre/creatis_noise/case%d/case%d_T%d0_noise.npy"%(case,case,T),noise_img)


        img = img.astype(np.int16)
        mu = 0
        sigma = 10
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        np.save("E:/dir_lab_pre/creatis_noise/case%d/case%d_T%d0_gasuss.npy"%(case,case,T),img)
