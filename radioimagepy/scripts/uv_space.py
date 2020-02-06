import numpy as np
from scipy import constants


class uv_space:
    '''
    Container class to holt the content of a uv-file according to the fits-idi standard.
    For the Stokes parameter, the convention RR,LL,RL,LR is thought to be used
    in the uv-file!

    Parameters to init
    ------------------
    data: fits data object
            'PRIMARY' part of the uv-data, that contains the visibilitys.
    freq: Skalar in unit Hz
            Frequency, the data is taken at (i.e. data['PRIMARY'].header['CRVAL4'])
    IF_count: Skalar
            Number of IFs, that should be used (number of IFs stored in
            data['PRIMARY'].header['NAXIS5'])
    u_ind: Skalar
            Row-index of the u-component, usually 0
    v_ind: Skalar
            Row-index of the v-component, usually 1

    Returns
    -------
    uv_space-object

    Properties
    ----------
    get_uv_plane:
            Returns an array of u and v coordinates.
    get_IFs:
            Returns a dictonary, containing an array of Stokes-I
            values per IF.

    Functions
    ---------
    uvplot:
            Takes an matplotlib axes object and plots the uv-plot in it.
    '''
    def __init__(self, data, freq, IF_count=1, u_ind=0, v_ind=1):
        uv_plane = []

        # Get Wavelength for difmap-convenience
        lam = constants.c/freq

        # Get all uv-points for each visibility, the factor 2.998*10**8 is used to convert
        # light-seconds to metre
        for row in data:
            # Every visibility corresponds to two uv-points!
            uv_plane.append(np.array([row[0]*2.998*10**8/lam, row[1]*2.998*10**8/lam]))
            uv_plane.append(np.array([-row[0]*2.998*10**8/lam, -row[1]*2.998*10**8/lam]))

        # Get I-Stokes for all IFs and visibilities
        IF = {}
        for i in range(IF_count):
            I = []
            for row in data:
                matrice = row[7][0][0][i][0]

                RR = matrice[0]
                LL = matrice[1]
                #RL = matrice[2]
                #LR = matrice[3]

                row_I = (LL+RR)/2

                # Append them twice, since every visibility corresponds to two uv-points!
                I.append(row_I)
                I.append(row_I)
            IF['I_%.0f'%i] = np.array(I)

        self._uv_plane = np.array(uv_plane)
        self._IF = IF

    @property
    def get_uv_plane(self):
        '''
        Returns an array of u and v coordinates.
        '''
        return self._uv_plane

    @property
    def get_IFs(self):
        '''
        Returns a dictonary, containing an array of Stokes-I
        values per IF. Each array contains N rows, where
        [Real, Imag, Weigth] of each visibility are stored.
        '''
        return self._IF

    def uvplot(self, ax):
        '''
        Takes an matplotlib axes object and plots the uv-plot in it.

        Parameters
        ----------
        ax: Matplotlib axes object
                The figure, the plot is build in.

        Returns
        -------
        ax: Matplotlib axes object
                The figure, now with uv-plot, labels and inverted xaxis
                (do produce similar results as difmap)
        '''
        ax.plot(self._uv_plane[:,0], self._uv_plane[:,1], 'kx', markersize=1)
        ax.invert_xaxis()
        ax.set_xlabel(r'u/$\lambda$')
        ax.set_ylabel(r'v/$\lambda$')
        return ax