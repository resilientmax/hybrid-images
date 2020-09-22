import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import glob

def FFT(img):
    f = fft2(img)
    fshift = fftshift(f)
    return fshift

def invFFT(fimg):
    return ifft2(ifftshift(fimg))

def GaussianFitler(numRows, numCols, sigma, highPass=False):
    
    centerX = numRows / 2
    centerY = numCols / 2
    
    coeff = np.zeros((numRows, numCols))
    for i in range(numRows):
        for j in range(numCols):
            coeff[i][j] = np.exp(-((i - centerX)**2 + (j - centerY)**2) / (2 * sigma**2))
    GS_filter = 1 - coeff if highPass else coeff
    
    return GS_filter

def low_pass(lowPassImg, sigma):
    rows, columns = lowPassImg.shape[0], lowPassImg.shape[1]
    filter = GaussianFitler(rows, columns, sigma, False)
    f_Img = FFT(lowPassImg)
    return f_Img * filter

def high_pass(highPassImg, sigma):
    rows, columns = highPassImg.shape[0], highPassImg.shape[1]
    filter = GaussianFitler(rows, columns, sigma, True)
    f_Img = FFT(highPassImg)
    return f_Img * filter

def hybridImg(highPassImg, lowPassImg, sigmaHigh, sigmaLow):
    highPassed = high_pass(highPassImg, sigmaHigh)
    lowPassed = low_pass(lowPassImg, sigmaLow)
    return np.real(invFFT(highPassed + lowPassed))


def myHybridPipeline(fileNo):
    names = glob.glob('images/'+ fileNo + '?.*')
    fileName1 = names[0]
    fileName2 = names[1]
    img1 = cv2.imread(fileName1)
    img1 = cv2.resize(img1, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.imread(fileName2)
    img2 = cv2.resize(img2, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)

    sigmaHigh, sigmaLow = 20, 15
    hybrid = np.zeros(img1.shape)
    hybrid[:, :, 0] = hybridImg(img1[:,:,0], img2[:,:,0], sigmaHigh, sigmaLow)
    hybrid[:, :, 1] = hybridImg(img1[:,:,1], img2[:,:,1], sigmaHigh, sigmaLow)
    hybrid[:, :, 2] = hybridImg(img1[:,:,2], img2[:,:,2], sigmaHigh, sigmaLow)

    hybrid = hybrid.astype(int) 
    cv2.imwrite("output/" + fileNo + ".jpg",hybrid)
    hyb = cv2.imread("output/" + fileNo + ".jpg")
    horinzontal = np.concatenate((img1, img2), axis=1)
    final = np.concatenate((horinzontal, hyb), axis=1)
    cv2.imshow("Hybrid Image " + fileNo, final)
    cv2.waitKey(0)

for each in range(1,11):
    myHybridPipeline(str(each))
cv2.destroyAllWindows()