import LightPipes
from Wavefront import Wavefront
import matplotlib.pyplot as plt
import torch





wl = 1064e-9
n = 1
m = 1
size = 10e-3
N = 2**8

wf = Wavefront(pixelNumber=N, Lfield = size, wl = 1064e-9)
F=LightPipes.Begin(size,wl,N)
F= LightPipes.GaussBeam(F, w0 = 1e-3, LG=True, n=n, m=m)
print(F.grid_size)
print(F.wavelength)
print(F.N)


# wf.field[:] = torch.sqrt(torch.from_numpy(LightPipes.Intensity(F)))*torch.exp(1j*torch.from_numpy(LightPipes.Phase(F)))
# plt.imshow(torch.angle(wf.field))

# plt.show()


# plt.imshow(LightPipes.Intensity(F))
# plt.show()

# F = 