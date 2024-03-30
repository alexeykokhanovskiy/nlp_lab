import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 

class Wavefront():
    def __init__(self, pixelNumber = 2**10, Lfield = 5e-3, wl = 1030e-9):
        ## Field Grid parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pixelNumber = pixelNumber
        self.Lfield = Lfield 
        self.pixelSize = Lfield/pixelNumber
        self.wl = wl
        #Initialization of the field grids in real and reciprocal spaces
        self.scale  = torch.arange(-self.Lfield/2, self.Lfield/2, self.pixelSize)
        self.fscale  = torch.arange(-1/(2*self.pixelSize), 1/(2*self.pixelSize), 1/self.Lfield)
        self.X , self.Y  = torch.meshgrid(self.scale , self.scale , indexing='xy')
        self.X  = self.X .to(self.device)
        self.Y  = self.Y .to(self.device)
        self.FX , self.FY  = torch.meshgrid(self.fscale , self.fscale , indexing='xy')

        #Initialization of the field distributions in real and reciprocal spaces
        self.field  = torch.ones_like(self.X , dtype = torch.complex128)
        self.field  = self.field.to(self.device, dtype=torch.complex128)
        self.fourier  = self.F (self.field )
        self.fourier  = self.fourier.to(self.device, dtype=torch.complex128)

        self.k  = 2*torch.pi/wl
        self.kx  = 2*torch.pi*self.FX 
        self.kx  = self.kx.to(self.device, dtype=torch.complex128)
        self.ky  = 2*torch.pi*self.FY 
        self.ky  = self.ky.to(self.device, dtype=torch.complex128)
        self.kz   = torch.fft.fftshift(torch.sqrt(self.k **2 - self.kx **2 - self.ky **2))
        self.kz  = self.kz.to(self.device, dtype=torch.complex128)

    def F(self, field):
        return (torch.fft.fft2(field))
    
    def iF(self, field):
        return (torch.fft.ifft2(field))
    
    def Gauss(self, beamFwhm):
        self.field[:] = torch.exp(-2*torch.log(2)*((self.X /beamFwhm)**2 + (self.Y /beamFwhm)**2)) 
        self.fourier[:]= self.F (self.field ) 

    def propagate_angular_spec(self, distance):
        prop_factor = torch.exp(1j*distance*self.kz )
        prop_factor = prop_factor.to(self.device, dtype=torch.complex128)
        self.fourier[:] *= prop_factor 
        self.field[:] = self.iF(self.fourier )  

    def ThinLens (self, focalLength):
        transCoeff = torch.exp(-1j*self.k *(self.X **2+self.Y **2)/2/focalLength)
        self.field[:] = self.field  * transCoeff
        self.fourier[:] = self.F (self.field )

    def Vortex(self,Radius):
        self.field[:] *= torch.exp(1j*(torch.arctan2(self.Y , self.X )))*(torch.sqrt(self.X **2 + self.Y **2) < Radius)
        self.fourier[:] = self.F (self.field )

    def Gauss(self, beamFwhm):
        self.field[:] = torch.exp(-2*torch.log(torch.tensor(2))*((self.X /beamFwhm)**2 + (self.Y /beamFwhm)**2)) 
        self.fourier[:] = self.F (self.field ) 
        return torch.exp(-2*torch.log(torch.tensor(2))*((self.X /beamFwhm)**2 + (self.Y /beamFwhm)**2)) 
    
    def propagate_angular_spec(self, distance):
        prop_factor = torch.exp(1j*distance*self.kz)
        self.fourier[:] *= prop_factor 
        self.field[:] = self.iF(self.fourier) 

    def get_field_phase(self):
        return torch.angle(self.field)

    def CircularAperture(self, Radius):
        self.field[:] *= torch.sqrt(self.X **2 + self.Y **2) < Radius
        self.fourier[:] = self.F (self.field )

    def BlockAperture(self, Radius):
        self.field[:] *= torch.sqrt(self.X **2 + self.Y **2) > Radius
        self.fourier[:] = self.F (self.field )

    def ThinLens(self, focalLength):
        transCoeff = torch.exp(-1j*self.k *(self.X **2+self.Y **2)/2/focalLength)
        transCoeff = transCoeff.to(self.device)
        self.field[:] = self.field  * transCoeff
        self.fourier[:] = self.F (self.field )

    def DiffractionGrating(self, focalLength, x0, y0):
        transCoeff = torch.exp(-1j*2*torch.tensor(torch.pi)*(self.X *x0 + self.Y *y0)/torch.tensor(self.wl)/focalLength)
        self.field[:] = self.field  * transCoeff
        self.fourier[:] = self.F (self.field )
    
    def BlockDiffraction(self,Radius, focalLength, x0, y0):
        DiffPattern = torch.angle(torch.exp(-1j*2*torch.tensor(torch.pi)*(self.X *x0 + self.Y *y0)/torch.tensor(self.wl)/focalLength))
        DiffPattern[:] *= torch.sqrt(self.X **2 + self.Y **2) < Radius
        self.field[:] = torch.abs(self.field)*torch.exp(1j*(torch.angle(self.field )+ DiffPattern)) 
        self.fourier[:] = self.F (self.field)

    def load_field(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.pixelNumber, self.pixelNumber), interpolation=cv2.INTER_CUBIC) 
        image = 255 - torch.from_numpy(image)
        self.field[:] = image/torch.max(image)
        self.fourier[:] = self.F(self.field)
    
    def get_intensity(self):
        return torch.abs(torch.Tensor.cpu(wf.field))
    
    def get_k_space(self):
        return torch.abs(torch.Tensor.cpu(torch.fft.fftshift(wf.fourier)))

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



if __name__ == "__main__":
    wf = Wavefront(pixelNumber= 2**9, Lfield = 20e-3)
    wf.Gauss(beamFwhm=10e-3)
    plt.imshow(wf.get_intensity())
    plt.show()
    # wf.propagate_angular_spec(distance=10e-3)
    # phase, err = retrieve_phase_GerberSaxton_mod(wf.field, 3 , 'cuda')
    # plt.imshow(torch.abs(phase))
    # plt.show()

    # wf.propagate_angular_spec(distance=10e-2)
    # plt.imshow(wf.get_intensity())
    # plt.imshow(wf.get_k_space())
    # plt.show()

    # wf.Gauss(beamFwhm=4e-3)
    # wf.ThinLens(focalLength=10e-3)
    # wf.propagate_angular_spec(distance=10e-3)
    # plt.show()
    # wf.CircularAperture(Radius=2e-4)

    # wf.DiffractionGrating(10e-3, 1e-3, 0)
    # wf.BlockAperture(Radius=2e-4)
    # wf.BlockDiffraction(2e-4, 10e-3, 1e-3, 0)
    # # wf.ThinLens (focalLength=25e-3)
    # # wf.propagate_angular_spec (25e-3)
    # plt.subplot(121) 
    # plt.title('Real space')
    # plt.imshow(torch.angle(torch.Tensor.cpu(wf.field )), cmap='jet')
    # plt.subplot(122)
    # plt.title('K-space')
    # plt.imshow(torch.abs(torch.Tensor.cpu((torch.fft.fftshift(wf.fourier )))), cmap='jet')

    # # plt.imshow(torch.abs(torch.Tensor.cpu(torch.fft.fftshift(wf.fourier ))), cmap='jet')

    # plt.show()




    # plt.imshow(torch.abs(torch.Tensor.cpu(wf.field )))

    # wf.Vortex(2e-4)
    # wf.propagate_angular_spec (distance=40e-3)
    # plt.imshow(torch.abs(torch.Tensor.cpu(wf.field )))
    # plt.show()


    # plt.subplot(211)
    # plt.imshow(torch.abs(wf.field ))
    # plt.subplot(212)
    # plt.imshow(torch.angle(wf.field ))


    # plt.show()
   
    # print(wf.field .size())
    
    # start_time = time.time()


    
    # field = torch.fft.ifft2(torch.fft.fft2(wf.field ))
    # plt.imshow(torch.abs(field))
    # plt.show()
    # wf.ThinLens (10e-3)
    # wf.Vortex(10e-3)

    # wf.propagate_angular_spec (distance=40e-3)
    # wf.Gauss(beamFwhm=4e-4)
    # plt.imshow(wf.get_field_phase())
    # plt.show()
    # start_time = time.time()
    # for i in range(20):
    #     wf.propagate_angular_spec(distance=20e-3)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('Elapsed time_numpy: ', elapsed_time)
    # print(wf.field[0,0])


    # start_time = time.time()
    # for i in range(100):
    #     wf.propagate_angular_spec (distance=20e-3)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('Elapsed time: ', elapsed_time)

    # # wf.ThinLens (focalLength= 10e-3)
    # wf.propagate_angular_spec (distance=20e-3)
    # # plt.imshow(torch.abs(wf.field ))
    # plt.imshow(torch.abs(wf.field ))
    # # plt.imshow(torch.fft.fftshift(torch.abs(wf.fourier )))
    # plt.show()