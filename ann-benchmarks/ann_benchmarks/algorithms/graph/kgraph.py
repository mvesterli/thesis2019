from __future__ import absolute_import
import os
import numpy
import pykgraph
from ann_benchmarks.algorithms.base import BaseANN

# benchmark k-nn graph construction
class KGraph(BaseANN):
    def __init__(self, metric, params):
        if type(metric) == unicode:
            metric = str(metric)
        self.name = 'KGraph(%s)' % (metric)
        self._metric = metric
        self._L = params['L']
        self._recall = params['recall']


    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        self._kgraph = pykgraph.KGraph(X, self._metric)
        self._kgraph.build(
            reverse = 0,
            K = self._count,
            # L must always be > count
            L = self._count+self._L,
            recall = self._recall)

    def query(self, idx, n):
        # The graph contains more than k neighbors per node, but they seem to be in sorted order.
        return self._kgraph.get_nn(idx)[0][:self._count]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'KGraph(l=%d, recall=%.2f)' % (self._L, self._recall)

