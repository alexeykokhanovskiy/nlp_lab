import torch
import matplotlib.pyplot as plt

from Wavefront import Wavefront   

wf = Wavefront()

def retrieve_phase_GerberSaxton_mod (field_input, Niter, device):
    # phase = torch.ones_like(field_input, dtype = torch.complex128)
    field_input = field_input.to(device)
    phase = torch.rand(field_input.size()[0],field_input.size()[1], dtype = torch.complex128)
    phase = phase.to(device)
    err = np.zeros(Niter)
    AmpConstrait = torch.abs(field_input)
    AmpConstrait = AmpConstrait.to(device)
    FourierConstrait = torch.abs(torch.fft.fft2(field_input))
    FourierConstrait = FourierConstrait.to(device)

    for i in range(0,Niter):
        phase[:] = AmpConstrait*torch.exp(1j*torch.angle(phase))
        phase[:] = torch.fft.fft2(phase)
        phase[:] = FourierConstrait*torch.exp(1j*torch.angle(phase))
        phase[:] = torch.fft.ifft2(phase)
        err[i] = torch.sqrt(torch.sum((torch.abs(field_input) - torch.abs(phase))**2))
        # print(f'Iteration {i}: Error {err[i]}')
    return torch.Tensor.cpu(phase), err




