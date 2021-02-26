import timeit

setup_numpy = '''
import numpy as np
s = ""
for i in range(500):
    s += '0101'
'''

print(min(timeit.Timer("np.fromstring(s, 'i1') - 48", setup=setup_numpy).repeat(7, 10000)))
print(min(timeit.Timer("(np.fromstring(s, 'i1') - 48).astype('float')", setup=setup_numpy).repeat(7, 10000)))
print(min(timeit.Timer("np.fromstring(s,'u1') - ord('0')", setup=setup_numpy).repeat(7, 10000)))
print(min(timeit.Timer("(np.fromstring(s,'u1') - ord('0')).astype('float')", setup=setup_numpy).repeat(7, 10000)))
# print(min(timeit.Timer("np.array(list(map(int, s)))", setup=setup_numpy).repeat(7, 1000)))
# print(min(timeit.Timer("np.array(list(s), dtype='float')", setup=setup_numpy).repeat(7, 1000)))

# (np.fromstring(s,'u1') - ord('0')).astype('float')
# np.fromstring(s, 'i1') - 48
