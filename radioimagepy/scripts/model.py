import importlib.util
spec = importlib.util.spec_from_file_location("utils", '/home/rune/Schreibtisch/radioimagepy/radioimagepy/scripts/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

import numpy as np
from numpy.fft import irfft2, rfft2, fftshift


class Model:
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
        """
        Returns model as image.
        """
        return self._model

    def convolve(self, beam):
        beam_fft = rfft2(beam)
        mod_fft = rfft2(np.copy(self._model))
        return fftshift(irfft2(mod_fft*beam_fft))