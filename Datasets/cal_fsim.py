import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2, ifftshift

def FeatureSIM(imageRef, imageDis):
    rows, cols = imageRef.shape[:2]
    I1 = np.ones((rows, cols))
    I2 = np.ones((rows, cols))
    Q1 = np.ones((rows, cols))
    Q2 = np.ones((rows, cols))

    if len(imageRef.shape) == 3: # images are colorful
        Y1 = 0.299 * imageRef[:,:,0] + 0.587 * imageRef[:,:,1] + 0.114 * imageRef[:,:,2]
        Y2 = 0.299 * imageDis[:,:,0] + 0.587 * imageDis[:,:,1] + 0.114 * imageDis[:,:,2]
        I1 = 0.596 * imageRef[:,:,0] - 0.274 * imageRef[:,:,1] - 0.322 * imageRef[:,:,2]
        I2 = 0.596 * imageDis[:,:,0] - 0.274 * imageDis[:,:,1] - 0.322 * imageDis[:,:,2]
        Q1 = 0.211 * imageRef[:,:,0] - 0.523 * imageRef[:,:,1] + 0.312 * imageRef[:,:,2]
        Q2 = 0.211 * imageDis[:,:,0] - 0.523 * imageDis[:,:,1] + 0.312 * imageDis[:,:,2]
    else: # images are grayscale
        Y1 = imageRef
        Y2 = imageDis

    Y1 = np.double(Y1)
    Y2 = np.double(Y2)

    minDimension = min(rows,cols)
    F = max(1,round(minDimension / 256))
    aveKernel = np.ones((F, F)) / (F * F)

    aveI1 = convolve2d(I1, aveKernel, mode='same')
    aveI2 = convolve2d(I2, aveKernel, mode='same')
    I1 = aveI1[::F,::F]
    I2 = aveI2[::F,::F]

    aveQ1 = convolve2d(Q1, aveKernel, mode='same')
    aveQ2 = convolve2d(Q2, aveKernel, mode='same')
    Q1 = aveQ1[::F,::F]
    Q2 = aveQ2[::F,::F]

    aveY1 = convolve2d(Y1, aveKernel, mode='same')
    aveY2 = convolve2d(Y2, aveKernel, mode='same')
    Y1 = aveY1[::F,::F]
    Y2 = aveY2[::F,::F]

    PC1 = phasecong2(Y1)
    PC2 = phasecong2(Y2)

    # Define kernels
    dx = np.array([[3, 0, -3],
                   [10, 0, -10],
                   [3, 0, -3]]) / 16

    dy = np.array([[3, 10, 3],
                   [0, 0, 0],
                   [-3, -10, -3]]) / 16

    # Perform convolution on Y1
    IxY1 = convolve2d(Y1, dx, mode='same')
    IyY1 = convolve2d(Y1, dy, mode='same')
    gradientMap1 = np.sqrt(IxY1 ** 2 + IyY1 ** 2)

    # Perform convolution on Y2
    IxY2 = convolve2d(Y2, dx, mode='same')
    IyY2 = convolve2d(Y2, dy, mode='same')
    gradientMap2 = np.sqrt(IxY2 ** 2 + IyY2 ** 2)

    # Define fixed thresholds
    T1 = 0.85
    T2 = 160

    # Calculate PCSimMatrix
    PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1 ** 2 + PC2 ** 2 + T1)

    # Calculate gradientSimMatrix
    gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + T2) / (gradientMap1 ** 2 + gradientMap2 ** 2 + T2)

    # Calculate PCm
    PCm = np.maximum(PC1, PC2)

    # Calculate FSIM
    SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
    FSIM = np.sum(SimMatrix) / np.sum(PCm)

    # Define fixed thresholds
    T3 = 200
    T4 = 200

    # Calculate ISimMatrix
    ISimMatrix = (2 * I1 * I2 + T3) / (I1 ** 2 + I2 ** 2 + T3)

    # Calculate QSimMatrix
    QSimMatrix = (2 * Q1 * Q2 + T4) / (Q1 ** 2 + Q2 ** 2 + T4)

    # Define lambda
    lambda_val = 0.03

    # Calculate SimMatrixC
    SimMatrixC = gradientSimMatrix * PCSimMatrix * ((ISimMatrix * QSimMatrix) ** lambda_val) * PCm
    FSIMc = np.sum(SimMatrixC) / np.sum(PCm)

    return FSIM, FSIMc

# Define phasecong2 function (equivalent implementation in Python)
def phasecong2(im):
    nscale = 4
    norient = 4
    minWaveLength = 6
    mult = 2
    sigmaOnf = 0.55
    dThetaOnSigma = 1.2
    k = 2.0
    epsilon = 0.0001

    thetaSigma = np.pi / norient / dThetaOnSigma

    rows, cols = im.shape
    imagefft = fft2(im)  # Fourier transform of image

    zero = np.zeros((rows, cols))
    EO = [[None for _ in range(norient)] for _ in range(nscale)]  # Array of convolution results.

    estMeanE2n = []
    ifftFilterArray = [None for _ in range(nscale)]  # Array of inverse FFTs of filters

    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    lp = lowpassfilter([rows, cols], 0.45, 15)  # Radius 0.45, 'sharpness' 15

    logGabor = [None for _ in range(nscale)]

    for s in range(nscale):
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] = logGabor[s] * lp
        logGabor[s][0, 0] = 0

    spread = [None for _ in range(norient)]

    for o in range(norient):
        angl = (o - 1) * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    EnergyAll = np.zeros((rows, cols))
    AnAll = np.zeros((rows, cols))

    for o in range(norient):
        sumE_ThisOrient = zero
        sumO_ThisOrient = zero
        sumAn_ThisOrient = zero
        Energy = zero
        for s in range(nscale):
            filter = logGabor[s] * spread[o]
            ifftFilt = np.real(ifft2(filter)) * np.sqrt(rows * cols)
            ifftFilterArray[s] = ifftFilt
            EO[s][o] = ifft2(imagefft * filter)
            An = np.abs(EO[s][o])
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(EO[s][o])
            sumO_ThisOrient += np.imag(EO[s][o])
            if s == 0:
                EM_n = np.sum(filter ** 2)
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for s in range(nscale):
            E = np.real(EO[s][o])
            O = np.imag(EO[s][o])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        medianE2n = np.median(np.reshape(np.abs(EO[0][o]) ** 2, (1, rows * cols)))
        meanE2n = -medianE2n / np.log(0.5)
        estMeanE2n.append(meanE2n)

        noisePower = meanE2n / EM_n

        EstSumAn2 = zero
        for s in range(nscale):
            EstSumAn2 += ifftFilterArray[s] ** 2

        EstSumAiAj = zero
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                EstSumAiAj += ifftFilterArray[si] * ifftFilterArray[sj]

        sumEstSumAn2 = np.sum(EstSumAn2)
        sumEstSumAiAj = np.sum(EstSumAiAj)

        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj

        tau = np.sqrt(EstNoiseEnergy2 / 2)
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = np.sqrt((2 - np.pi / 2) * tau ** 2)

        T = EstNoiseEnergy + k * EstNoiseEnergySigma
        T = T / 1.7

        Energy = np.maximum(Energy - T, zero)
        EnergyAll += Energy
        AnAll += sumAn_ThisOrient

    ResultPC = EnergyAll / AnAll
    return ResultPC


def lowpassfilter(sze, cutoff, n):
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('Cutoff frequency must be between 0 and 0.5')

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be an integer >= 1')

    if len(sze) == 1:
        rows = sze
        cols = sze
    else:
        rows = sze[0]
        cols = sze[1]

    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)  # A matrix with every pixel = radius relative to center.
    f = 1 / (1 + (radius / cutoff) ** (2 * n))  # The filter
    f = ifftshift(f)
    return f
