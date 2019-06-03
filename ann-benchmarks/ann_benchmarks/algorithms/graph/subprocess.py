from __future__ import absolute_import

from ann_benchmarks.data import float_unparse_entry
from ann_benchmarks.algorithms.subprocess import Subprocess

def FloatSubprocess(args, params):
    return Subprocess(args, float_unparse_entry, params, True)
