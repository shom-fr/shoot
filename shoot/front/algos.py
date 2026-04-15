# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter
import math
from math import floor, ceil
from numpy.fft import fft2
import xoa.coords as xcoords
import matplotlib.pyplot as plt


#%% 
#### CCA ALGO #####


def getFrontInWindow(
    w,
    head,
    minTheta,
    minPopProp,
    minPopMeanDiff,
    minSinglePopCohesion,
    minGlobalPopCohesion,
    corners,
):
    """
    This functions detects fronts in slidding windows. If a front is detected, the function will return
    2 1D arrays (x and y) with the coordinate values corresponding to the location of the front.
    """

    # empty arrays de xdata, ydata e z
    xdata, ydata = np.array([]), np.array([])
    exitType = 0

    # mask is an array with the same shape of w, that is 1 if in that index position w = np.nan and 0 otherwise
    mask = np.isnan(w).astype('int')
    haveNaNs = np.any(mask[:]).astype(
        'int'
    )  # haveNaNs=1 if mask has 1s (that correspond to NaNs in matrix w)
    n_NaNs = 0

    if haveNaNs:
        n_NaNs = sum(mask.flatten()[:])  # count nr of 1s (NaNs in matrix w) that there are
        #print("percent nans : %.2f"%(n_NaNs / len(w.flatten())))
        if n_NaNs / len(w.flatten()) > 0.5:  # window can't have more than 50% of its pixels as NaNs
            exitType = -1
            return None, None, None, exitType

    mi_ma = [np.nanmin(w), np.nanmax(w)]  # array with minimum and maximum value of w
    n = ceil((mi_ma[1] - mi_ma[0]) / 0.02)  # number of bins
    bins = np.arange(mi_ma[0], mi_ma[1], 0.02)  # to define the bins sequence
    [y, xout] = np.histogram(w[:], bins, mi_ma)  # y->frequency counts, Xout->bin location
    xout = np.mean(
        np.vstack([xout[0:-1], xout[1:]]), axis=0
    )  # xout to be relative to the centers of the bins

    try:
        thresValue = xout[0]
    except:
        thresValue = 0

    totalCount = len(w.flatten()) - n_NaNs  # nr of non NaN pixels
    threshPopACount, threshSeparation, threshPopAMean, threshPopBMean = 0, -1, 0, 0

    w[mask == 1] = 0  # Replace NaNs with 0's (when mask is 1 replace values of array w for 0)
    totalSum = sum(w.flatten())  # sum of values of matrix w
    totalSumSquares = sum(w.flatten() * w.flatten())  # sum of the squares of the values of w

    # In this for loop we are going to discover which line is going to make the best separation between the average
    # of population on the left and on the right (A and B) - and that is going to be the thresValue
    for k in range(1, n - 1):  # ignore the first and last candidates (senão seria de 0 a n)
        popASum = sum(y[0 : k + 1] * xout[0 : k + 1])
        popBSum = sum(y[k + 1 :] * xout[k + 1 :])
        popACount = sum(y[0 : k + 1])  # sum of frequencies (y) from populationA
        popBCount = sum(y[k + 1 :])  # sum of frequencies (y) from populationB

        popAMean = popASum / popACount
        try:  # to avoid the zerodivisionerror that was poping up
            popBMean = popBSum / popBCount
        except ZeroDivisionError:
            popBMean = 0
        separation = popACount * popBCount * (popAMean - popBMean) * (popAMean - popBMean)
        if separation > threshSeparation:
            threshSeparation = separation
            thresValue = xout[k]
            threshPopACount = popACount
            threshPopAMean = popAMean
            threshPopBMean = popBMean
            
    if thresValue < 1e-1 : #we have detected the frontiers
        exitType = -1
        return None, None, None, exitType
        
    # abort in case the proportion of population A is less that a certain minimum OR in case the proportion of population B is less that a certain minimum
    #print("PopProp : %.2f"%(threshPopACount / totalCount))
    if (threshPopACount / totalCount < minPopProp) or (
        1.0 - threshPopACount / totalCount < minPopProp
    ):
        exitType = 1
        return None, None, None, exitType

    # abort this window if the difference in the populations means is less than a minimum value
    #print("MeanDiff : %.2f"%(threshPopBMean - threshPopAMean))
    if threshPopBMean - threshPopAMean < minPopMeanDiff:
        exitType = 2
        return None, None, None, exitType

    # Calculate the criterion function THETA (TAUopt) in page 72 of the paper
    totalMean = totalSum / totalCount
    variance = totalSumSquares - (totalMean * totalMean * totalCount)
    theta = threshSeparation / (variance * totalCount)
    #print("theta : %.2f"%theta)
    if theta < minTheta:  # abort if theta is lower than a certain minimum
        exitType = 3
        return None, None, None, exitType

    # Cohesion - now that we know the separation value. Based on this value we will check the matrix element by
    # element, and check whether is bigger or lower than the separation
    # we check if it's bigger bellow or to the right (when its bigger we add from one side, when its lower add to the other)
    # Count the nr of times a population A cell is immediately adjacent to another popA cell and the same for popB
    # A cell can be adjacent on 4 sides. Count only 2 of them (bottom and right side) because doing all 4 would be
    # redundant. Do not count diagonal neighbors
    countANextToA, countBNextToB, countANextToAOrB, countBNextToAOrB = 0, 0, 0, 0
    [n_rows, n_cols] = w.shape
    for col in range(0, n_cols - 1):
        for row in range(0, n_rows - 1):
            if haveNaNs & (mask[row, col] | mask[row + 1, col] | mask[row, col + 1]):
                continue

            # examine the bottom neighbor
            if w[row, col] <= thresValue:  # if matrix pixel < than the element of separation
                countANextToAOrB = countANextToAOrB + 1  # increase by 1 countANextToAOrB
                if w[row + 1, col] <= thresValue:  # if pixel of bottom row < than separation
                    countANextToA = countANextToA + 1  # increase countANextToA
            else:  # if pixel > than separation
                countBNextToAOrB = countBNextToAOrB + 1  # increase countBNextToAOrB
                if w[row + 1, col] > thresValue:  # if pixel of bellow row > separation
                    countBNextToB = countBNextToB + 1  # increase countBNextToB

            # Examine the right neighbor
            if w[row, col] <= thresValue:  # if matrix pixel < separation
                countANextToAOrB = countANextToAOrB + 1  # increase countANextToAOrB
                if w[row, col + 1] <= thresValue:  # if right pixel < separation
                    countANextToA = countANextToA + 1  # increase countANextToA
            else:  # if matrix pixel > separation
                countBNextToAOrB = countBNextToAOrB + 1  # increase countBNextToAOrB
                if w[row, col + 1] > thresValue:  # if right pixel > separation
                    countBNextToB = countBNextToB + 1  # increase countBNextToB

    popACohesion = countANextToA / countANextToAOrB
    popBCohesion = countBNextToB / countBNextToAOrB
    globalCohesion = (countANextToA + countBNextToB) / (countANextToAOrB + countBNextToAOrB)
    
    # print("popAcohes : %.2f"%popACohesion)
    # print("popBcohes : %.2f"%popBCohesion)
    # print("globalCohes : %.2f"%globalCohesion)


    # These ifs are in case of errors (parameters below certain limits)
    if (
        (popACohesion < minSinglePopCohesion)
        or (popBCohesion < minSinglePopCohesion)
        or (globalCohesion < minGlobalPopCohesion)
    ):
        exitType = 4
        return None, None, None, exitType

    # OK if we reach here we have a front. Compute its contour
    X = np.linspace(head[0], head[1], n_cols)
    Y = np.linspace(head[2], head[3], n_rows)
    if corners.size == 0:
        w = w.astype('double')
        if haveNaNs:
            w[w == 0] = np.nan  # Need to restore the NaNs to not invent new contours around zeros
        c = plt.contour(
            X, Y, w, [thresValue]
        )  # Create and store a set of contour lines or filled regions.
    else:
        # the 4 corners have these indices [17,32,17,32; 17,32,1,16; 1,16,1,16;1,16,17,32]
        # and the variable corners has one of its rows (the current to be retained sub-window)

        X = X[np.arange(corners[2] - 1, corners[3])]
        Y = Y[np.arange(corners[0] - 1, corners[1])]
        w = w[
            np.arange(corners[0], corners[1]).min()
            - 1 : np.arange(corners[0], corners[1]).max()
            + 1,
            np.arange(corners[2], corners[3]).min()
            - 1 : np.arange(corners[2], corners[3]).max()
            + 1,
        ]

        if haveNaNs:
            w[w == 0] = np.nan  # Need to restore the NaNs to not invent new contours around zeros

        if (np.isnan(w)).all() == True:
            c = np.array([])
        else:
            c = plt.contour(
                X, Y, w, [thresValue]
            )  # Create and store a set of contour lines or filled regions.

    # breakpoint()
    try:
        M = c.allsegs[
            :
        ]  # list of arrays for contour c. Each array corresponds to a line that may or may not be drawn. This list can have any number of arrays
    except:
        M = []

    M = [x for x in M if x]  # if the list has empty arrays we will drop them

    count = 0  # to iterate through the various arrays

    # Create list of booleans (True or False) wether the conditions bellow are fulfilled
    # Each array (line of contour) must have more that 7 data points and they can't be closed lines
    lista = []
    for i in range(len(M[:])):
        lista.append(
            [(len(x) < 7 or (x[0][0] == x[-1][0] and x[0][1] == x[-1][1])) for x in M[:][i]]
        )

        # if False the line will be drawn
        # if True the line will be ignored

    for value in lista:
        if value == [True]:
            continue  # return to the top of the for loop
        else:
            # For the first array of M we will take all the values of x and put them into an array
            x = [(M[:][count][0][i][0]).round(4) for i in range(len(M[:][count][0]))]

            # For the first array of M we will take all the values of y and put them into an array
            y = [(M[:][count][0][i][1]).round(4) for i in range(len(M[:][count][0]))]

            # save the x and y data points for each line in an xdata and ydata array
            xdata, ydata = np.append(xdata, x), np.append(ydata, y)

            count += 1

    z = thresValue

    if xdata.size == 0:
        exitType = 5

    return xdata, ydata, z, exitType


def cca_sied(temp, minPopProp=0.2, minPopMeanDiff=0.4, minTheta=0.7, minSinglePopCohesion=0.9, minGlobalPopCohesion =0.7):
    
    """
    This function applies the Cayula-Cornillon Algorithm Single Image Edge Detector (CCA_SIED) to a single image
    with data from an xarray.
    For a single image, the function return the fronts coordinates (x,y) points
    """

    # convert the latitude and longitude columns to a numpy array
    lat = xcoords.get_lat(temp).values
    lon = xcoords.get_lon(temp).values

    lat_min, lat_max, lon_min, lon_max = lat.min(), lat.max(), lon.min(), lon.max()

    X, Y = np.meshgrid(lon, lat)  # create rectangular grid out of two given 1D arrays

    lat = Y.T
    lon = X.T

    Z = temp.values

    head = np.array([lon_min, lon_max], dtype='float64')
    head = np.append(head, [lat_min, lat_max])

    z_dim = Z.shape  # dimensions/shape of matrix Z (rows, cols)

    z_actual_range = np.array(
        [np.nanmin(Z[:]), np.nanmax(Z[:])]
    )  # range of data (minimum and maximum of matrix Z)
    nx = z_dim[1]  # number of columns of matrix Z
    ny = z_dim[0]  # number of rows of matrix Z
    node_offset = 0

    # index 4 -> minimum value of Z; index5 -> maximum value of Z; index6 -> node_offset=0
    head = np.append(head, np.array([z_actual_range[0], z_actual_range[1], node_offset]))
    head = np.append(head, np.array((head[1] - head[0]) / (nx - int(not node_offset))))
    head = np.append(head, np.array((head[3] - head[2]) / (ny - int(not node_offset))))

    # cayula parameters;
    #minPopProp-> minimum proportion of each population
    #minPopMeanDiff ->minimum difference between the means of the 2 populations
    # minPopProp, minPopMeanDiff, minTheta, minSinglePopCohesion, minGlobalPopCohesion = (
    #     0.2,
    #     0.4,
    #     0.7,
    #     0.9,
    #     0.7,
    # )

    [n_rows, n_cols] = z_dim  # nr of rows and nr of columns of matrix Z
    winW16, winW32, winW48 = 16, 32, 48

    # arrays that will store the contour of every front that will be detected
    xdata_final, ydata_final = np.array([]), np.array([])

    s = 0  # s=1 means subwindows do NOT share a common border. With s = 0 they do.
    xSide16 = winW16 * head[7]
    ySide16 = winW16 * head[8]
    xSide32 = (winW32 - s) * head[7]
    ySide32 = (winW32 - s) * head[8]

    nWinRows = floor(n_rows / winW16)  # times a window can slide over the rows
    nWinCols = floor(n_cols / winW16)  # times a window can slide over the columns

    for wRow in range(1, nWinRows - 1):
        # start and stop indices and coords of current window
        r1 = (wRow - 1) * winW16 + 1
        r2 = r1 + winW48 - s

        y0 = head[2] + (wRow - 1) * ySide16

        for wCol in range(1, nWinCols - 1):
            c1 = (wCol - 1) * winW16 + 1
            c2 = c1 + winW48 - s
            x0 = head[0] + (wCol - 1) * xSide16
            wPad = Z[r1 - 1 : r2, c1 - 1 : c2]  # 49x49 (or 48x48 if s == 1) Window

            rr = np.array([1, 1, 2, 2])
            cc = np.array([1, 2, 2, 1])

            if s == 1:
                corners = np.array(
                    [[17, 32, 17, 32], [17, 32, 1, 16], [1, 16, 1, 16], [1, 16, 17, 32]]
                )  # less good
            else:
                corners = np.array(
                    [[17, 33, 17, 33], [17, 33, 1, 17], [1, 17, 1, 17], [1, 17, 17, 33]]
                )

            for k in range(
                0, 4
            ):  # loop over the 4 slidding 32X32 sub-windows of the larger 48x48 one
                m1 = (rr[k] - 1) * winW16 + 1
                m2 = m1 + 2 * winW16 - s  # indices of the slidding 33X33 window
                n1 = (cc[k] - 1) * winW16 + 1
                n2 = n1 + 2 * winW16 - s

                w = wPad[m1 - 1 : m2, n1 - 1 : n2].astype('double')  # sub window with size 33x33

                # corners coordinates
                subWinX0 = x0 + (cc[k] - 1) * xSide16
                subWinX1 = subWinX0 + xSide32
                subWinY0 = y0 + (rr[k] - 1) * ySide16
                subWinY1 = subWinY0 + ySide32

                R = np.array([subWinX0, subWinX1, subWinY0, subWinY1])
                xdata, ydata, z, exitType = getFrontInWindow(
                    w,
                    R,
                    minTheta,
                    minPopProp,
                    minPopMeanDiff,
                    minSinglePopCohesion,
                    minGlobalPopCohesion,
                    corners[k, :],
                )
                if exitType == 0:

                    xdata_final = np.append(xdata_final, xdata)
                    ydata_final = np.append(ydata_final, ydata)

    return xdata_final, ydata_final

def cca_bf(temp,minPopProp = 0.2, minPopMeanDiff = 0.4, minTheta = 0.7, minSinglePopCohesion=0.9,  minGlobalPopCohesion=0.7, njump = 10):
    """
    This function applies the Cayula-Cornillon Algorithm Single Image Edge Detector based on brut force
    windows analysis.
    For a single image, the function return the fronts coordinates (x,y) points
    """
    lat = xcoords.get_lat(temp).values
    lon = xcoords.get_lon(temp).values
    temp = temp.values
    xdata_final = np.array([])
    ydata_final = np.array([])
    
    ny, nx = temp.shape
    mask = np.isnan(temp)
    maxima = np.empty((0, 2), dtype=np.int64)
    minima = np.empty((0, 2), dtype=np.int64)
    wx = 32
    wy = 32
    wx2 = wx // 2
    wy2 = wy // 2
    for j in range(1, ny - 1,njump):
        for i in range(1, nx - 1,njump):
            if mask[j, i]:
                continue
            i0 = max(0, i - wx2)
            i1 = min(nx, i + wx2 + 1)
            j0 = max(0, j - wy2)
            j1 = min(ny, j + wy2 + 1)
            
            if mask[j - 1 : j + 2, i - 1 : i + 2].all():
                continue
            
            wtemp = temp[j0 : j1, i0 : i1]
            if np.sum(np.isnan(wtemp))/len(wtemp.flatten()) > minPopProp: 
                continue
            R = np.array([lon[i0 : i1].min(), lon[i0 : i1].max(), lat[j0 : j1].min(), lat[j0 : j1].max()]) 
            xdata, ydata, z, exitType = getFrontInWindow(
                wtemp,
                R,
                minTheta,
                minPopProp,
                minPopMeanDiff,
                minSinglePopCohesion,
                minGlobalPopCohesion,
                np.array([]),
            )
            if exitType == 0:
                xdata_final = np.append(xdata_final, xdata)
                ydata_final = np.append(ydata_final, ydata)
                    
    return xdata_final, ydata_final 


def wrapper_cca(temp, minPopProp=0.2, minPopMeanDiff=0.4, minTheta=0.7, minSinglePopCohesion=0.9, minGlobalPopCohesion =0.7, algo = 'sied', njump = 10):
    """
    Function that calculates the fronts matrix. Given an image (SST data respective to one day) it applies the
    Cayula-Cornillon Algorithm for Single Image Edge Detection (CCA-SIED) to discover the fronts.
    It returns the matrix with the fronts: if pixel = 1 it was considered a front, otherwise, pixel = 0
    It basically converts the (x,y) coordinate points to indexes of the frontal probability matrix. These indexes are considered fronts
    """

    lat = xcoords.get_lat(temp)
    lon = xcoords.get_lon(temp)

    lat_dims = temp.sizes[lat.dims[0]]
    lon_dims = temp.sizes[lon.dims[0]]

    lat_max = lat.values.max()
    lat_min = lat.values.min()
    lon_max = lon.values.max()
    lon_min = lon.values.min()

    div_rows = (lat_max - lat_min) / (lat_dims-1)
    div_cols = (lon_max - lon_min) / (lon_dims-1)

    fp = np.zeros((lat_dims, lon_dims))  # initialize a matrix of zeros

    # 2 empty arrays that will store the x and y values of the lines that are suposed to be drawn
    x = np.array([])
    y = np.array([])

    if algo == 'sied' : 
        xdata_final, ydata_final = cca_sied(temp, minPopProp=minPopProp, minPopMeanDiff=minPopMeanDiff, minTheta=minTheta, minSinglePopCohesion=minSinglePopCohesion, minGlobalPopCohesion =minGlobalPopCohesion)
    else : 
        xdata_final, ydata_final = cca_bf(temp, minPopProp=minPopProp, minPopMeanDiff=minPopMeanDiff, minTheta=minTheta, minSinglePopCohesion=minSinglePopCohesion, minGlobalPopCohesion =minGlobalPopCohesion,njump=njump)
    x = np.append(x, xdata_final)
    y = np.append(y, ydata_final)
    
    cols_x = np.array([])
    for value in x:  # convert values in array x to the respective index in a (1001, 1401) matrix
        aux_x = (-lon_min + value) / div_cols  # these numbers are relative to the MUR data
        cols_x = np.append(cols_x, aux_x)

    rows_y = np.array([])
    for value in y:  # convert values in array y to the respective index in a (1001, 1401) matrix
        aux_y = (lat_max - value) / div_rows  # these numbers are relative to the MUR data
        rows_y = np.append(rows_y, aux_y)

    cols_x = np.round(cols_x)
    rows_y = np.round(rows_y)
        
    for i in range(len(cols_x)):  # it could also be len(rows_y)
        fp[int(rows_y[i]), int(cols_x[i])] = fp[int(rows_y[i]), int(cols_x[i])] + 1

    fp[fp != 0] = 1
    return np.flipud(fp), xdata_final, ydata_final

#%%
#### BOA ALGO #####


def filt5(lon, lat, ingrid, nodata=np.nan):
    """
    Find peaks in 5 directions. Flag as 5
    Finds maximum of a 5x5 sliding window. If the central pixel is the maximum, this is flagged as a one.
    All other pixels are flagged as zero.
    """

    nodatidx = (
        ingrid.flatten() * np.nan
    )  # creates 1D array with as much values as the matrix ingrid, with NANs
    outgrid = np.zeros(ingrid.shape)  # outgrid is a matrix with the shape of ingrid, full of Zeros

    l1 = len(lat)
    l2 = len(lon)

    for i in range(3, l1 - 1):
        for j in range(3, l2 - 1):
            subg = ingrid[
                (i - 3) : (i + 2), (j - 3) : (j + 2)
            ]  # return the last 5 rows of the last 5 columns of the matrix
            if np.isnan(subg).sum() == 25:  # if all values in submatrix subg are null values:
                outgrid[i, j] = 0
            else:
                vec = np.array(subg).T.flatten()  # array with values of the transpose subg matrix
                ma = np.argmax(subg.flatten())  # index with the maximum value of subg array
                mi = np.argmin(subg.flatten())  # index with the minimum value of subg array

                if (
                    ma == 12 or mi == 12
                ):  # if ma or mi is the middle value of 5X5 matrix (if the central pixel is the maximum)
                    outgrid[i - 1, j - 1] = 1  # flagged as 1
                else:
                    outgrid[i - 1, j - 1] = 0  # all other pixels are flagged as 0

    return outgrid


def filt3(lon, lat, ingrid, grid5):
    """
    Find peaks in 3 directions. FLag as 3
    Returns a median smoothed grid of satellite data
    """

    outgrid = ingrid * 0  # matrix of 0s with shape of ingrid matrix
    l1 = len(lat)
    l2 = len(lon)

    for i in range(3, l1 - 1):
        for j in range(3, l2 - 1):
            if grid5[i, j] == 0:
                subg = ingrid[(i - 2) : (i + 1), (j - 2) : (j + 1)]  # submatrix subg (3x3)
                if (
                    np.isnan(subg).sum() == 9
                ):  # if all values in submatrix subg (3x3) are null values:
                    outgrid[i - 1, j - 1] = ingrid[i - 1, j - 1]
                else:
                    vec = np.array(
                        subg
                    ).T.flatten()  # array with values of the transpose subg matrix
                    ma = np.argmax(subg.flatten())  # index with the maximum value of subg array
                    mi = np.argmin(subg.flatten())  # index with the minimum value of subg array

                    if ma == 4 or mi == 4:  # if ma or mi is the middle value of 3X3 matrix
                        outgrid[i - 1, j - 1] = np.nanmedian(subg)  # median while ignoring NaNs.
                    else:
                        outgrid[i - 1, j - 1] = ingrid[i - 1, j - 1]

            else:
                outgrid[i - 1, j - 1] = ingrid[i - 1, j - 1]

    return outgrid


def boa(lon, lat, ingrid, nodata=np.nan, direction=False):

    def filter2(x, filt):
        """
        Workhorse filter from EBImage. Modified so we don't need colorspace and other annoying requirements
        """

        dx = x.shape
        df = filt.shape

        if (df[0] // 2 == 0) or (df[1] // 2 == 0):
            sys.exit('dimensions of "filter" matrix must be odd')
        if (dx[0] < df[0]) or (dx[1] < df[1]):
            sys.exit("dimensions of 'x' must be bigger than 'filter'")

        cx = tuple(elem // 2 for elem in dx)
        cf = tuple(elem // 2 for elem in df)

        wf = np.zeros(shape=dx)  # matrix with zeros with shape of x

        wf[cx[0] - cf[0] - 1 : cx[0] + cf[0], cx[1] - cf[1] - 1 : cx[1] + cf[1]] = (
            filt  # put values of filt in middle of matrix wf
        )

        wf = fft2(wf)  # apply the 2 dimensional discrete fourier transform

        dim_x = np.array(dx[0:2])
        dim_x = np.append(dim_x, math.prod(dx) / math.prod(dx[0:2]))

        aux1 = np.arange(cx[0], dx[0] + 1)
        aux2 = np.arange(1, cx[0])
        index1 = np.concatenate((aux1, aux2), axis=None)
        index1 = index1 - 1

        aux3 = np.arange(cx[1], dx[1] + 1)
        aux4 = np.arange(1, cx[1])
        index2 = np.concatenate((aux3, aux4), axis=None)
        index2 = index2 - 1
        # this indices will be used to reorder values of matrix y

        y = (scipy.fft.ifft2(scipy.fft.fft2(x) * wf)).real

        y = np.array([[y[i][j] for j in index2] for i in index1])

        return y

    # ======================================================#
    # Main BOA algorithm
    # ======================================================#
    gx = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # filter in x
    gy = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # filter in y

    np.nan_to_num(
        ingrid, nan=-9999, posinf=-9999, neginf=-9999
    )  # replace NaN and inf values with -9999

    grid5 = filt5(lon, lat, ingrid, nodata=nodata)
    grid35 = filt3(lon, lat, ingrid, grid5)

    # make an index of bad values and land pixels.
    grid35 = grid35.astype("float")
    grid35[grid35 == -9999] = np.nan
    naidx = np.isnan(grid35)  # matrix with shape of grid35 (True if value is nan, False otherwise)
    # convert these (True values of naidx) to zeros (in grid35) for smoothing purposes
    grid35[naidx] = 0

    # perform the smoothing (Sobel filter)
    tgx = filter2(grid35, gx)
    tgy = filter2(grid35, gy)

    tx = tgx / np.nansum(abs(np.array(gx).flatten()))
    ty = tgy / np.nansum(abs(np.array(gy).flatten()))
    front = np.sqrt((tx**2) + (ty**2))

    # ======================================================#
    # landmask and edge dilation
    # ======================================================#

    land = naidx * 1
    land = land.astype("float")

    land[land == 1] = np.nan
    land[~np.isnan(land)] = 1

    # ======================================================#
    # landmask and edge dilation using raster!
    # ======================================================#

    l2 = lon.size
    l1 = lat.size

    midx = land * np.nan

    midx[5 : (l1 - 2), 5 : (l2 - 2)] = 1

    land = np.multiply(land, midx)

    ssr = np.flip(front.T, 0)

    # Apply a sliding window kernell to the land matrix
    mask = scipy.signal.convolve2d(
        np.flip(land.T, 0),
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3, 3),
        boundary='symm',
        mode='same',
    )

    matrix_front = mask * np.flip(
        front.T, 0
    )  # matrix of mask raster file * matrix of ssr raster file

    if direction == True:
        #   ************************************
        #   *** Calculate Gradient Direction ***
        #   ************************************

        n = ingrid.size  # nr of elements of the grid matrix
        grid_shape = ingrid.shape

        GRAD_DIR = np.zeros(n)  # matrix of zeros with shape of ingrid matrix

        for i in range(n):
            GRAD_DIR[i] = math.atan2(tgy.flatten()[i], tgx.flatten()[i])

        GRAD_DIR = GRAD_DIR * 180 / math.pi  # change radians to degrees

        OK = np.where(GRAD_DIR < 0)

        OK = np.array(OK)

        if OK.size > 1:
            GRAD_DIR[OK] = 360 - abs(
                GRAD_DIR[OK]
            )  # Adjust to 0-360 scheme (make negative degrees positive)

        GRAD_DIR = (
            360 - GRAD_DIR + 90
        ) % 360  # Convert degrees so that 0 degrees is North and East is 90 degrees
        GRAD_DIR = GRAD_DIR.reshape(grid_shape)

        grad_dir = np.flip(GRAD_DIR.T, 0)

        # create array grdir (result from multiplication of grad_dir_matrix and mask_matrix (its the conv matrix))
        grdir_matrix = np.flip(GRAD_DIR.T, 0) * mask

        dic = {'grdir': grdir_matrix, 'front': matrix_front}

    else:
        matrix_front

    return matrix_front


def boa_wrapper(temp, threshold=0.3):
    """
    Function to, for a given xarray with a longitude, latitude and SST columns,
    identifies fronts through the application of BOA algorithm.
    We also need to define a threshold value to later get the frontal probabilities matrix
    (if the pixel value is greater than the threshold, then it is considered a front, otherwise its not).
    """

    lat = xcoords.get_lat(temp).values
    lon = xcoords.get_lon(temp).values
    ingrid = temp.values

    front = boa(lon=lon, lat=lat, ingrid=ingrid, nodata=np.nan, direction=False)
    front = np.flip(front, axis=0)
    front = np.array(
        [[front[j][i] for j in range(len(front))] for i in range(len(front[0]) - 1, -1, -1)]
    )

    front = np.where(front >= threshold, 1, front)
    front = np.where(front < threshold, 0, front)

    return np.flipud(front)


#%% 
#### CANNY ALGO ####

def compute_gradients(image):
    # image doit être en niveaux de gris
    # Sobel pour gradient horizontal (dx=1, dy=0)
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # Sobel pour gradient vertical (dx=0, dy=1)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude du gradient
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # Orientation du gradient (en radians)
    orientation = np.arctan2(Gy, Gx)

    return Gx, Gy, magnitude, orientation

def non_max_suppression(mag, angle):
    H, W = mag.shape
    Z = np.zeros((H, W), dtype=np.float64)

    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):

            q = 255
            r = 255

            # angle 0°
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # angle 45°
            elif (22.5 <= angle[i, j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # angle 90°
            elif (67.5 <= angle[i, j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # angle 135°
            elif (112.5 <= angle[i, j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if mag[i, j] >= q and mag[i, j] >= r:
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0

    return Z

def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    high = img.max() * high_ratio
    low  = high * low_ratio

    H, W = img.shape
    res = np.zeros((H, W), dtype=np.uint8)

    strong = 255
    weak = 50

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j     = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j]     = weak

    return res, weak, strong

def hysteresis(img, weak=50, strong=255):
    H, W = img.shape

    for i in range(1, H-1):
        for j in range(1, W-1):
            if img[i, j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img

def my_canny_from_gradients(Gx, Gy):
    mag = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx)

    nms = non_max_suppression(mag, angle)
    dt, weak, strong = double_threshold(nms)
    edges = hysteresis(dt, weak, strong)

    return edges

def canny_front(temp, tmin = None, tmax = None, sigma=5, apertureSize=5):
    """
    This code is extracted from https://github.com/CoLAB-ATLANTIC/JUNO/
    Function that receives a dataframe with SST data relative to a certain day and returns the front matrix
    obtained due to the aplication of the Canny algorithm.
    For each image a Gaussian filter (with a certain sigma value) might be applied (depending on the data)
    Tmin and Tmax are the limits of the threshold and apertureSize is the size of the Sobel operator (default=3X3)
    """

    # get temp values 
    temp = temp.values

    # Convert the temperature values to the uint8 format with values between 0-255
    temp_day = ((temp - np.nanmin(temp)) * (1 / (np.nanmax(temp) - np.nanmin(temp)) * 255)).astype(
        'uint8'
    )
    
    if not tmin : 
        gx = cv2.Sobel(temp_day, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(temp_day, cv2.CV_64F, 0, 1)
        grad = np.sqrt(gx**2 + gy**2)
        
        tmax = np.nanpercentile(grad, 95)
        tmin = 0.4 * tmax


    temp_day = np.flipud(
        temp_day
    )  # flipud -> Reverse the order of elements along axis 0 (up/down).

    # if its MUR data we have to apply gaussian filter with certain sigma value (~5)
    temp_day = gaussian_filter(temp_day, sigma=sigma)

    # apply the canny algorithm from OpenCV
    canny = cv2.Canny(temp_day, tmin, tmax, L2gradient=False, apertureSize=apertureSize)

    return np.flipud(canny)
