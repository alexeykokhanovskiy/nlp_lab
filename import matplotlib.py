from Wavefront import Wavefront
from RetrievePhase import *
import matplotlib.pyplot as plt 


wf = Wavefront()
wf.Gauss(beamFwhm=1e-3)
wf.Axicon(kp = 10000)
wf.propagate_angular_spec(distance=10e-2)


target_mask = wf.get_intensity() 
wf.Gauss(beamFwhm=1e-3)
input_intensity = wf.get_intensity()
initial_phase = torch.randn(input_intensity.shape)
spatial_field = torch.sqrt(input_intensity) * torch.exp(1j * initial_phase)

uniform_amplitude = torch.sqrt(target_intensity)
uniform_amplitude /= uniform_amplitude.max()
# plt.imshow(wf.get_intensity())
# plt.show()