
n = 1e7

flop = n**3 # floating point operations scale roughly with O(n^3) when solving dense linear systems

pc_flops = 50e12 # RTX 4090
sc_flops = 1742000e12 # Frontier supercomputer @ Oak Ridge National Labs

sc_seconds = flop / sc_flops
pc_seconds = flop / pc_flops

print('supercomputer seconds: ', sc_seconds)
print('supercomputer hours: ', sc_seconds / 3600)
print('supercomputer days: ', sc_seconds / 3600 / 24)
print('supercomputer years: ', sc_seconds / 3600 / 24 / 365)

print('laptop seconds: ', pc_seconds)
print('laptop hours: ', pc_seconds / 3600)
print('laptop days: ', pc_seconds / 3600 / 24)
print('laptop years: ', pc_seconds / 3600 / 24 / 365)
