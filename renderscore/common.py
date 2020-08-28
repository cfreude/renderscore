import os
import numpy as np


def cache_to_disk(_path):

    if not os.path.exists(_path):
        print 'Creating cache dir %s' % _path
        os.mkdir(_path)

    def decorator(_func):

        def helper(*_args):
            cache_path = os.path.join(_path, '%s_%d.npy' % (_func.__name__, hash(_args)))

            try:
                result = np.load(cache_path)
                print 'Loaded cache %s' % cache_path
            except IOError:
                print 'Cache not found %s' % cache_path
                result = _func(*_args)
                np.save(cache_path, result)

            return result

        return helper

    return decorator


def luminance(rgb_img):
    dim = rgb_img.shape
    lf = [0.27, 0.67, 0.06]  # [0.2126, 0.7152, 0.0722]
    to_lum = np.tile(lf, [dim[0], dim[1], 1])
    return np.sum(to_lum * rgb_img, axis=2, dtype=np.float64)


def tone_map(hdr_image, method='reinhard', gamma=2.2):

    hdr_image_lum = luminance(hdr_image)
    delta = 0.001

    tmp = np.log(delta + hdr_image_lum)
    l_average = np.exp(np.mean(tmp))
    l_white = np.mean(hdr_image_lum[hdr_image_lum > np.quantile(hdr_image_lum, 0.999)])

    key_a = np.max([0.0, 1.5 - 1.5 / (l_average * 0.1 + 1.0)]) + 0.1

    scaled = (key_a / l_average) * hdr_image

    ldr_image = np.zeros(hdr_image.shape, dtype=np.float64)

    method_dict = {
        'log': 0,
        'linear': 1,
        'reinhard': 2,
        'reinhard_mod': 3,
        'naive': 4,
        'lum': 5,
        None: 6,
    }

    if method_dict[method] == 0:  # log
        ldr_image = np.log(hdr_image + 0.001)

    elif method_dict[method] == 1:  # linear
        max_lum = np.max(hdr_image_lum)
        ldr_image = hdr_image / max_lum

    elif method_dict[method] == 2:  # reinhard
        ldr_image = scaled / (1.0 + scaled)

    elif method_dict[method] == 3:  # reinhard_mod
        ldr_image = (scaled * (1.0 + scaled / np.power(l_white, 2.0))) / (1.0 + scaled)

    elif method_dict[method] == 4:  # naive
        ldr_image = hdr_image / (hdr_image + 1.0)

    elif method_dict[method] == 5:  # lum
        ldr_image = luminance(hdr_image)

    elif method_dict[method] == 6:  # none
        ldr_image = hdr_image

    if gamma != 1.0:
        ldr_image = np.power(ldr_image, 1.0/gamma)

    return ldr_image


class OnlineStats:

    # https://www.johndcook.com/blog/standard_deviation/

    def __init__(self, _dtype=np.float):
        self.n = 0
        self.oldM = 0
        self.newM = 0
        self.oldS = 0
        self.newS = 0

        self.dtype = _dtype
        self.minimum = None
        self.maximum = None

    def clear(self):
        self.__init__()

    def push(self, x):

        x = np.array(x, dtype=self.dtype)

        self.n += 1

        # See Knuth TAOCP vol 2, 3rd edition, page 232

        if self.n == 1:
            self.oldM = self.newM = x
            self.oldS = np.zeros(x.shape, dtype=self.dtype)
            self.minimum = self.maximum = x
        else:
            self.newM = self.oldM + (x - self.oldM) / self.n
            self.newS = self.oldS + (x - self.oldM) * (x - self.newM)
            self.minimum = np.minimum(self.minimum, x)
            self.maximum = np.maximum(self.maximum, x)

        # set up for next iteration
        self.oldM = self.newM
        self.oldS = self.newS

    def num_data_values(self):
        return self.n

    def mean(self):
        if self.n > 0:
            return self.newM
        else:
            try:
                return np.zeros(self.newM.shape, self.dtype)
            except AttributeError:
                return 0

    def variance(self, doff=1):

        if self.n > 1:
            return self.newS / float(self.n - doff)
        else:
            try:
                return np.zeros(self.newM.shape, self.dtype)
            except AttributeError:
                return 0

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def max(self):
        return self.maximum

    def min(self):
        return self.minimum


if __name__ == '__main__':
    pass
