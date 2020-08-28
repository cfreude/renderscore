import numpy as np
from collections import OrderedDict

from common import OnlineStats
from render_score import score_func_array


class DistanceBase(object):

    def __init__(self):
        self.ref_mean = None
        self.ref_std = None
        self.ref_coov = None

    def set_reference(self, _ref_mean, _ref_std):
        self.ref_mean = _ref_mean
        self.ref_std = _ref_std
        self.ref_coov = np.copy(self.ref_std)
        np.divide(self.ref_std, self.ref_mean, out=self.ref_coov, where=self.ref_mean > 0)

    def get_values(self):
        return OrderedDict([('ReferenceMean', self.ref_mean), ('ReferenceCoov', self.ref_coov)])


class SampleStats(DistanceBase):

    def __init__(self):
        super(SampleStats, self).__init__()
        self.stats = OnlineStats()

    def push(self, _sample):
        self.stats.push(_sample)

    def get_values(self):
        cofv = np.copy(self.stats.standard_deviation())
        np.divide(self.stats.standard_deviation(), self.stats.mean(), out=cofv, where=self.stats.mean() > 0)
        return OrderedDict([('SampleMean', self.stats.mean()), ('SampleCoov', cofv)])


class RenderScore(SampleStats):

    def __init__(self):
        super(RenderScore, self).__init__()

    def get_values(self):
        if self.ref_mean is None:
            rs = None
        else:
            rs = score_func_array(self.stats.mean(), self.ref_mean, self.stats.standard_deviation(), self.ref_std)
        return OrderedDict([('RenderScore', rs)])


class RelMSE(SampleStats):

    def __init__(self):
        super(RelMSE, self).__init__()

    def push(self, _sample):
        d = _sample - self.ref_mean
        out = np.copy(d)
        np.divide(d, self.ref_mean, out=out, where=self.ref_mean > 0)
        self.stats.push(out)

    def get_values(self):
        return OrderedDict([('RelMSE', self.stats.mean())])
