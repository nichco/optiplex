import numpy as np



# c = np.array([0.00261664])

# ctol = 1e-2

# # if any(np.abs(c)) > ctol:
# #     print('test')


# if any(np.abs(c) > ctol):
#     print('test')
#     print(np.abs(c))
#     print(any(np.abs(c) > ctol))




old = np.array([1.0]) * 1e-1
new = np.array([1.00001]) * 1e-1

# absolute(a - b) <= (atol + rtol * absolute(b))

atol = 1e-9
rtol = 1e-2

is_equal = np.allclose(new, old, atol=atol, rtol=rtol)

print(is_equal)

print(atol + rtol * np.abs(old))