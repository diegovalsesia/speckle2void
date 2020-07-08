from keras.utils import Sequence

import numpy as np


class DataWrapper():
    """
    The N2V_DataWrapper extracts random sub-patches from the given data and manipulates 'num_pix' pixels in the
    input.

    Parameters
    ----------
    X          : array(floats)
                 The noisy input data. ('SZYXC' or 'SYXC')
    Y          : array(floats)
                 The same as X plus a masking channel.
    batch_size : int
                 Number of samples per batch.
    num_pix    : int, optional(default=1)
                 Number of pixels to manipulate.
    shape      : tuple(int), optional(default=(64, 64))
                 Shape of the randomly extracted patches.
    value_manipulator : function, optional(default=None)
                        The manipulator used for the pixel replacement.
    """

    def __init__(self, X, Y, batch_size, shape=(64, 64)):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape = shape
        self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape)
        self.dims = len(shape)
        self.n_chan = X.shape[-1]

        if self.dims == 2:
            self.patch_sampler = self.__subpatch_sampling2D__
            self.X_Batches = np.zeros([X.shape[0], shape[0], shape[1], X.shape[3]])
            self.Y_Batches = np.zeros([Y.shape[0], shape[0], shape[1], Y.shape[3]])

    def __len__(self):
        #return int(np.ceil(len(self.X) / float(self.batch_size)))
        return int(len(self.X) // float(self.batch_size))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]
        self.patch_sampler(self.X, self.Y, self.X_Batches, self.Y_Batches, idx, self.range, self.shape)


        return self.X_Batches[idx], self.Y_Batches[idx]

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
    
    @staticmethod
    def __subpatch_sampling2D__(X, Y, X_Batches, Y_Batches, indices, range, shape):
        for j in indices:
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_Batches[j] = X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]]
            Y_Batches[j] = Y[j, y_start:y_start + shape[0], x_start:x_start + shape[1]]

