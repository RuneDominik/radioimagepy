import numpy as np


def PCA(image):
    '''
    Compute the major components of an image. The Image is treated as a
    distribution.

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be used as distribution

    Returns
    -------
    cog_x: Skalar
            X-position of the distributions center of gravity
    cog_y: Skalar
            Y-position of the distributions center of gravity
    psi: Skalar
            Angle between first mjor component and y-axis

    '''
    pix_x, pix_y, image = im_to_array_value(image)

    cog_x = np.average(pix_x, weights=image)
    cog_y = np.average(pix_y, weights=image)

    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    cov = np.cov(delta_x, delta_y, aweights=image, ddof=1)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    width, length = np.sqrt(eig_vals)

    psi = np.arctan(eig_vecs[1, 1] / eig_vecs[0, 1])

    return cog_x, cog_y, width, length, psi


def im_to_array_value(image):
    '''
    Transforms the image to an array of pixel coordinates and the containt
    intensity

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be transformed

    Returns
    -------
    x_coords: Numpy 1Darray (N*M, 1)
            Contains the x-pixel-position of every pixel in the image
    y_coords: Numpy 1Darray (N*M, 1)
            Contains the y-pixel-position of every pixel in the image
    value: Numpy 1Darray (N*M, 1)
            Contains the image-value corresponding to every x-y-pair

    '''
    x_coords = []
    y_coords = []
    value = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            x_coords.append(x)
            y_coords.append(y)
            value.append(image[x, y])
    return np.asarray(x_coords), np.asarray(y_coords), np.asarray(value)


def array_to_im(shape, x, y, value):
    '''
    Transforms given x, y and pixel-value, so that the pixel-value
    is the content of the resulting pixel (x,y)

    Parameters
    ----------
    shape: Int
            Shape of the returned Image
    x: List or Numpy 1Darray (N,1)
            Contains the x-pixel-position of every value, that should be
            stored in the resulting image
    y: List or Numpy 1Darray (N, 1)
            Contains the y-pixel-position of every value, that should be
            stored in the resulting image
    value: List or Numpy 1Darray (N, 1)
            Contains the image-value corresponding to every x-y-pair

    Returns
    -------
    im: Numpy 2Darray (shape, shape)
            Image, given pixel-values are filled in, remaining image is
            set to value 0

    '''
    im = np.zeros((shape, shape))
    for i in range(len(x)):
        if im[x[i], y[i]] == 0:
            im[x[i], y[i]] = value[i]
        else:
            im[x[i], y[i]] = im[x[i], y[i]] + value[i]   
    return im

