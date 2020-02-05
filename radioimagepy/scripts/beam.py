import importlib.util
spec = importlib.util.spec_from_file_location("utils", '/home/rune/Schreibtisch/radioimagepy/radioimagepy/scripts/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

from astropy.convolution import Gaussian2DKernel
import numpy as np
from numpy.fft import irfft2, rfft2, fftshift


def beam_from_image(beam_im):
    # Supress sidelobes by cutting everything from the image, 
    # that is smaller than the FWHM of the central gauss
    beam_im[beam_im < 0.5*beam_im.max()] = 0

    # Image shapes
    x = beam_im.shape[1]
    y = beam_im.shape[0]

    # Innermost region of the Image
    x_left, x_right = (np.round((x/2)-100)).astype(int), np.round(((x/2)+100)).astype(int)
    y_left, y_right = (np.round((y/2)-100)).astype(int), np.round(((y/2)+100)).astype(int)

    # Get Beam-Parameters by PCA, might need to get inverted
    # x to y and vice versa
    cog_x, cog_y, BMIN, BMAJ, BPA = utils.PCA(beam_im[x_left:x_right, y_left:y_right])
    cog_x = np.round(cog_x).astype(int)
    cog_y = np.round(cog_y).astype(int)

    # Return beam params, 4 times the beam axis is to approx account for the FWHM
    # cut-of above, inverting the BPA is to aggre to difmap standards
    return x_left+cog_x, y_left+cog_y, 4*BMIN, 4*BMAJ, -BPA


class Beam:
    '''
    When BMIN, BMAJ and BPA from beam_from_image are used, than they need to 
    be transformed: 
    sigma_l = BMAJ/(2*np.sqrt(2*np.log(2)))
    sigma_w = BMIN/(2*np.sqrt(2*np.log(2)))
    theta = np.pi/2+BPA
    '''
    def __init__(self, sigma_l, sigma_w, theta, x_size, y_size):
        # Use astropy to initialize beam
        gaussian_2D_kernel = Gaussian2DKernel(sigma_l, sigma_w, theta, x_size=x_size, y_size=y_size)

        # Normalize beam, so peak has value 1
        gaussian_2D_kernel = gaussian_2D_kernel*(1/gaussian_2D_kernel.array.max())

        # Get only array as beam-param
        self._beam = np.copy(gaussian_2D_kernel.array)

    @property
    def get_beam(self):
        """
        Returns beam as image.
        """
        return self._beam

    def convolve(self, model):
        beam_fft = rfft2(np.copy(self._beam))
        mod_fft = rfft2(model)

        return fftshift(irfft2(mod_fft*beam_fft))