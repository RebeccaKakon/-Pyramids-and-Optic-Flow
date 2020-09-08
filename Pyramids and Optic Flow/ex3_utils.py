import sys
from typing import List
import math
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def isGrayScale(img):
    if (len(img.shape) < 3):
        return True
    else:
        return False


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
Given two images, returns the Translation from im1 to im2
:param im1: Image 1
:param im2: Image 2
:param step_size: The image sample size:
:param win_size: The optical flow window size (odd number)
:return: Original points [[y,x]...], [[dU,dV]...] for each points
 """
    if (not isGrayScale(im1)) and (not isGrayScale(im2)):
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    else:
        im1_gray = im1
        im2_gray = im2
    xy = []
    uv = []

    nigzeret = np.array([[1, 0, -1]])
    Ix = cv2.filter2D(im1_gray, -1, nigzeret, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im1_gray, -1, np.transpose(nigzeret), borderType=cv2.BORDER_REPLICATE)
    It = im2_gray - im1_gray

    w = win_size  # for me
    for i in range(w, im1.shape[0] - w + 1, step_size):
        for j in range(w, im1.shape[1] - w + 1, step_size):
            i_first, j_first = i - w // 2, j - w // 2
            i_last, j_last = i + w // 2 + 1, j + w // 2 + 1  # +1 !!
            b = -(It[i_first:i_last, j_first:j_last]).reshape(w * w, 1)
            temp = np.concatenate((Ix[i_first:i_last, j_first:j_last].reshape(w * w, 1),
                                   Iy[i_first:i_last, j_first:j_last].reshape(w * w, 1)), axis=1)
            A = np.asmatrix(temp)

            vxvy, e = np.linalg.eig(A.T * A)
            vxvy.sort()
            vxvy = vxvy[::-1]
            if vxvy[0] >= vxvy[1] > 1 and vxvy[0] / vxvy[1] < 100:
                Ab = np.array(np.dot(np.linalg.pinv(A), b))  # Ab = (A.T * A).I * A.T * b
                xy.append([(j + win_size / 2), (i + win_size / 2)])
                uv.append(Ab[::-1])

    # print("for shai ", np.mean(np.array(uv)))
    return np.array(xy), np.array(uv)


def conv1D(in_signal: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param kernel_size: 1-D array as a kernel
    :return: The convolved array
    """
    #!!!!!!!!!!!!!!!!!!!!!!!!this is your function !!!!!!!!!!!!!!!!!!!!!!!!!!

    if len(in_signal.shape) > 1:
        if in_signal.shape[1] > 1:
            raise ValueError("Input Signal is not a 1D array")
        else:
            in_signal = in_signal.reshape(in_signal.shape[0])

    inv_k = kernel_size[::-1].astype(np.float64)
    kernel_len = len(kernel_size)
    out_len = max(kernel_len, len(in_signal) + (kernel_len - 1))
    mid_kernel = kernel_len // 2
    padding = kernel_len - 1
    padded_signal = np.pad(in_signal, padding, 'constant')

    out_signal = np.ones(out_len)
    for i in range(out_len):
        st = i
        end = i + kernel_len

        out_signal[i] = (padded_signal[st:end] * inv_k).sum()

    return out_signal


def createGaussianKernel(k_size: int):
    #!!!!!!!!!!!!!!!!!!!your function!!!!!!!!!!!!!!!!!!!!
    if k_size % 2 == 0:
        raise ValueError("need to be zugi")

    k = np.array([1, 1], dtype=np.float64)
    iter_v = np.array([1, 1], dtype=np.float64)

    for i in range(2, k_size):
        k = conv1D(k, iter_v)
    k = k.reshape((len(k), 1))
    kernel = k.dot(k.T)
    kernel = kernel / kernel.sum()
    return kernel


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = createGaussianKernel(kernel_size)
    return cv2.filter2D(in_image, -1, kernel)


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    kernel_g = createGaussianKernel(5)
    copyImg = img
    pyr = [copyImg]
    for i in range(1, levels):  #going over the peramid
        copyImg = cv2.filter2D(copyImg, -1, kernel_g, borderType=cv2.BORDER_REPLICATE)
        copyImg = cv2.filter2D(copyImg, -1, cv2.transpose(kernel_g), borderType=cv2.BORDER_REPLICATE)
        copyImg = copyImg[::2, ::2]
        pyr.append(copyImg)

    return pyr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    expImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    if img.ndim == 3:
        expImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2, img.shape[2]), dtype=img.dtype)
    else:
        expImg = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=img.dtype)

    expImg[::2, ::2] = img  #expend every 2 pixel
    expImg = cv2.filter2D(expImg, -1, 2 * gs_k, cv2.BORDER_REPLICATE)
    expImg = cv2.filter2D(expImg, -1, cv2.transpose(2 * gs_k), cv2.BORDER_REPLICATE)

    return expImg


def resizeImgLikeTheMask(img: np.ndarray, mask: np.ndarray) -> (np.ndarray):
    j = mask.shape[0]
    i = mask.shape[1]
    img = cv2.resize(img, (i, j))
    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    img_1 = resizeImgLikeTheMask(img_1, mask)
    img_2 = resizeImgLikeTheMask(img_2, mask)

    naive = img_1 * mask + (1 - mask) * img_2  #naiv is the Formula we will do foe each level but now only for the imge given
    lap_ans = []  # the new lap pyr

    mask_pyr = gaussianPyr(mask, levels)  # we need this pyramid for the formula
    lap_img1 = laplaceianReduce(img_1, levels)  # the lap pyr for img1
    lap_img2 = laplaceianReduce(img_2, levels)  # the lapla pyr for img2

    for i in range(levels): #the general formula
        c = lap_img1[i] * mask_pyr[i] + (1 - mask_pyr[i]) * lap_img2[i]  # the function
        lap_ans.append(c)  # insert

    blend_pyr = laplaceianExpand(lap_ans)  # expand

    return naive, blend_pyr


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    kernel_g = createGaussianKernel(5)  # our kernel
    lap_pyr = []
    gaussianPy = gaussianPyr(img, levels)
    for i in range(1, len(gaussianPy)):
        #reduce the img to do the the lap pyramid thrw the formula
        exp = gaussExpand(gaussianPy[i], kernel_g)
        if exp.shape[0] != gaussianPy[i - 1].shape[0] and exp.shape[1] != gaussianPy[i - 1].shape[1]:
            exp = exp[:-1, :-1]

        exp = gaussianPy[i - 1] - exp
        lap_pyr.append(exp)
    lap_pyr.append(gaussianPy[levels - 1])

    return lap_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel = createGaussianKernel(5)  # sending to get our kernel to do the conv

    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        gauss_exp = gaussExpand(img, kernel) #expend the gauusian for the formula of lap
        if gauss_exp.shape[0] != lap_pyr[i].shape[0] and gauss_exp.shape[1] != lap_pyr[i].shape[1]:
            gauss_exp = gauss_exp[:-1, :-1]
        img = gauss_exp + lap_pyr[i]
    return img
