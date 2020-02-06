import importlib.util
spec = importlib.util.spec_from_file_location("utils", '/home/rune/Schreibtisch/radioimagepy/radioimagepy/scripts/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

import numpy as np
from numpy.fft import irfft2, rfft2, fftshift


class Model:
    '''
    Container class for a model image. Converts the content of an 
    difmap.MOD file to an actual image.

    Parameters
    ----------
    flux: Array or List (N,1)
            Flux values at position (x,y) or (r,theta)
    r: Array or List (N,1)
            Position of the model components in polar coordinates.
    theta: Array or List (N,1)
            Position angle of the model components in polar coordiantes.
    dx: Skalar 
            mas to px conversion factor for desired image along x-axis.
    dy: Skalar 
            mas to px conversion factor for desired image along y-axis.
    x_ref: Skalar
            Reference x-position of the image-center, i.e. radio core position.
    y_ref: Skalar
            Reference y-position of the image-center, i.e. radio core position.
    x-size: Skalar
            Desired x-dimension of the model image.
    y_size: Skalar
            Desired y-dimension of the model image, does nothing atm.

    Returns
    -------
    Model-object

    Properties
    ----------
    get_model:
            Returns the model-image as numpy 2Darray (x_size, y_size)

    Functions
    ---------
    convolve:
            Convolves model-image with another image
    '''
    def __init__(self, flux, r, theta, dx, dy, x_ref, y_ref, x_size, y_size):
        # Atm: Only supports quadratic images
        y_size=x_size

        # Difmap Mod-files store model points in polar coords - convert
        # them to kartesian in measures of px, shifted by ref-px
        x = np.round(r*np.cos(theta*np.pi/180)/dx) + x_ref
        y = np.round(r*np.sin(theta*np.pi/180)/dy) + y_ref

        # pop those points, that are outside image
        x_mask = (x > 0) & (x < x_size)
        y_mask = (y > 0) & (y < y_size)

        # Bitwise and to combine both
        combined_mask = x_mask & y_mask

        x = x[combined_mask]
        y = y[combined_mask]
        flux = flux[combined_mask]

        self._model = utils.array_to_im(x_size, x.astype(int), y.astype(int), flux)

    @property
    def get_model(self):
        '''
        Returns the model-image as numpy 2Darray (x_size, y_size)
        '''
        return self._model

    def convolve(self, beam):
        '''
        Convolves model-image with another image

        Parameters
        ----------
        model: Image or 2Darray (N,M)
                Image, that should be folded with model-image. Needs to be
                same dimension as model-image.

        Returns
        -------
        folded Image: 2Darray (N,M)
                Convolved and shifted Image.

        Raises
        ------
        None, so check dimensions of input Image!
        '''
        beam_fft = rfft2(beam)
        mod_fft = rfft2(np.copy(self._model))
        return fftshift(irfft2(mod_fft*beam_fft))