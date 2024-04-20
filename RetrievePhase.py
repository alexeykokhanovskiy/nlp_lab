import torch 
import numpy as np


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



def phase_retr_cycle(input_intensity, target_intensity, target_mask, iterations):
    """

    Parameters:
    input_intensity: The input intensity distribution.
    target_intensity: The desired intensity distribution in the Fourier plane.
    target_mask: A binary mask indicating the target region in the Fourier plane.
    iterations: Number of iterations to perform.

    Returns:
    phase_distribution: The phase distribution of the spatial field.
    final_intensity: The intensity distribution after applying the phase distribution.
    eta: Root-mean-square error over iterations.
    fourier_field: The Fourier field corresponding to the final spatial field.
    """
    # Initialize the spatial field with the initial phase and uniform amplitude.
    initial_phase = torch.randn(*input_intensity.shape)  # Random initial phase
    #initial_phase = fresnel_lens_phase_profile(target_intensity.shape, 50, 0.5)
    #initial_phase2 = linear_gradient_profile(target_intensity.shape, 10, 0)
    spatial_field = torch.sqrt(input_intensity) * torch.exp(1j * initial_phase)
    Nx, Ny = target_intensity.shape
    M = target_mask
    uniform_amplitude = torch.sqrt(target_intensity)
    uniform_amplitude /= uniform_amplitude.max()
    eta = []
    for k in range(iterations):
        # Forward propagation to the output plane.
        fourier_field = torch.fft.fft2(spatial_field)
        current_amplitude = torch.abs(fourier_field)
        current_amplitude /= torch.max(current_amplitude)
        alpha = 0  # Convergence parameter
        gamma = 1  # Noise suppressing parameter
        fourier_field = (M * (uniform_amplitude + alpha * (uniform_amplitude - current_amplitude)) +
                         gamma * (1 - M) * current_amplitude) * torch.exp(1j * torch.angle(fourier_field))
        # Backward propagation to the Fourier domain.
        spatial_field = torch.fft.ifft2(fourier_field)
        spatial_field = torch.sqrt(input_intensity) * torch.exp(1j * torch.angle(spatial_field))
        # Calculate root-mean-square error (RMSE)
        eta.append(torch.sqrt(torch.sum(torch.abs(current_amplitude**2 - uniform_amplitude**2)**2) /
                           torch.sum(torch.abs(uniform_amplitude**2)**2)).numpy())
    # Extract the final phase and intensity distributions.
    phase_distribution = torch.angle(spatial_field)
    final_intensity = torch.sqrt(input_intensity) 
    return phase_distribution, final_intensity, eta, fourier_field

if __name__ == '__main__':
    print('hello')
