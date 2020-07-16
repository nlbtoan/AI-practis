import cv2
import numpy as np

def get_translation(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def get_affine_cv(t, r, s):
    sin_theta = np.sin(r)
    cos_theta = np.cos(r)

    a_11 = s * cos_theta
    a_21 = -s * sin_theta

    a_12 = s * sin_theta
    a_22 = s * cos_theta

    a_13 = t[0] * (1 - s * cos_theta) - s * sin_theta * t[1]
    a_23 = t[1] * (1 - s * cos_theta) + s * sin_theta * t[0]
    return np.array([
        [a_11, a_12, a_13],
        [a_21, a_22, a_23]
    ])


img = cv2.imread('Exercise_1_input.png')

#height, width = img.shape[:2]
#res = cv2.resize(img, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)

#cv2.imshow('img', res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#A2 = get_affine_cv((tx, ty), angle, scale)
M = get_affine_cv((290, 50), 1, 1)
rows,cols,ch = img.shape
print(rows, cols);
dst = cv2.warpAffine(img, M, (600, 600))

cv2.imshow('img', dst)

#gaussian = cv2.GaussianBlur(img, (7, 9), 0)
#cv2.imshow('gaussion', gaussian)

#cv2.imwrite('gaussion.png', gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()

