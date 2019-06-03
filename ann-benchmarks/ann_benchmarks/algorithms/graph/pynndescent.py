from __future__ import absolute_import
import pynndescent
from ann_benchmarks.algorithms.base import BaseANN

class PyNNDescent(BaseANN):
    def __init__(self, metric, iterations):
        if metric == 'angular':
            metric = 'cosine'
        self._pynnd_metric = metric
        self._iterations = iterations

    def fit(self, X):
        nndescent = pynndescent.PyNNDescentTransformer(
            n_neighbors = self._count,
            metric      = self._pynnd_metric,
            n_iters     = self._iterations)
        self._graph = nndescent.fit_transform(X).tolil()

    def query(self, idx, n):
        return self._graph.rows[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'PyNNDescent(n_iters=%d)' % self._iterations
