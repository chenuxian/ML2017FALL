import sys
import numpy as np
from numpy import matlib, linalg
from skimage import transform, io

pixel = 120
img = np.zeros((pixel * pixel * 3, 415))
for i in range (415):
    file_name = sys.argv[1] + '/' + str(i) + ".jpg"
    tmp = io.imread(file_name)
    
    #resize
    tmp = transform.resize(tmp, (pixel, pixel, 3))

    img[:, i] = tmp.flatten()

img_mean = np.mean(img, axis=1)
for i in range (415):
    img[:, i] -= img_mean

A, B, C = np.linalg.svd(img, full_matrices=False)
eigenface = A[:, 0 : 4]

test_img = sys.argv[2]
img_number = int(test_img.replace(".jpg", ""))
weight = np.dot(np.transpose(img[:, img_number]), eigenface)

result = np.zeros((pixel * pixel * 3,))
for i in range(4):
    result += eigenface[:, i] * weight[i]
result += img_mean
result -= np.min(result)
result /= np.max(result)
result = (result * 255).astype(np.uint8)
result = result.reshape(pixel, pixel, 3)

io.imsave("reconstruction.jpg", result)
