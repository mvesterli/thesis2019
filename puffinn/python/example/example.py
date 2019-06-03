from _puffinnwrapper import *
import numpy
import time

d = 100
n = 100000
s = 1 << 27 # ~ 1mb

queries = 100

i = Index('angular', d, s)

for _ in range(n):
    i.insert([numpy.random.normal(0, 1) for _ in range(d)])

t0 = time.time()
i.rebuild()
print("Building index took %.2f seconds." % (time.time() - t0) )

t0 = time.time()
for _ in range(queries):
    i.search([numpy.random.normal(0, 1) for _ in range(d)], 10, 0.9)

print("Search the index took %.2f seconds." % (time.time() - t0))




