import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time

class Wavefront():
    def __init__(self, pixelNumber = 2**10, Lfield = 5e-3, wl = 1064e-9, device = 'cuda'):
        ## Field Grid parameters
        if device == "cuda":
            if torch.cuda.is_available(): 
                self.device = 'cuda'
            else:
                self.device = 'cpu' 
                print("GPU is unavailable")
        self.pixelNumber = pixelNumber
        self.Lfield = Lfield 
        self.pixelSize = Lfield/pixelNumber
        self.wl = wl
        #Initialization of the field grids in real and reciprocal spaces
        self.scale  = torch.arange(-self.Lfield/2, self.Lfield/2, self.pixelSize)
        self.fscale  = torch.arange(-1/(2*self.pixelSize), 1/(2*self.pixelSize), 1/self.Lfield)
        self.X, self.Y  = torch.meshgrid(self.scale, self.scale, indexing='xy')
        self.X = self.X.to(self.device)
        self.Y  = self.Y.to(self.device)
        self.FX, self.FY  = torch.meshgrid(self.fscale , self.fscale , indexing='xy')

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
        self.fourier[:]= self.F(self.field ) 

    def propagate_angular_spec(self, distance):
        prop_factor = torch.exp(1j*distance*self.kz )
        prop_factor = prop_factor.to(self.device, dtype=torch.complex128)
        self.fourier[:] *= prop_factor 
        self.field[:] = self.iF(self.fourier )  

    def ThinLens (self, focalLength):
        transCoeff = torch.exp(-1j*self.k*(self.X**2+self.Y**2)/2/focalLength)
        self.field[:] = self.field*transCoeff
        self.fourier[:] = self.F(self.field)

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
    
    def propagate_angular_spec_back(self, distance):
        prop_factor = torch.exp(-1j*distance*self.kz)
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
    
    def BlockAperture(self, Radius):
        self.field[:] *= torch.sqrt(self.X **2 + self.Y **2) > Radius
        self.fourier[:] = self.F (self.field )

    def ThinLens(self, focalLength):
        transCoeff = torch.exp(-1j*self.k*(self.X**2+self.Y**2)/2/focalLength)
        transCoeff = transCoeff.to(self.device)
        self.field[:] = self.field*transCoeff
        self.fourier[:] = self.F(self.field)
        return transCoeff

    def DiffractionGrating(self, focalLength, x0, y0):
        transCoeff = torch.exp(-1j*2*torch.tensor(torch.pi)*(self.X *x0 + self.Y *y0)/torch.tensor(self.wl)/focalLength)
        self.field[:] = self.field  * transCoeff
        self.fourier[:] = self.F (self.field )
    
    def BlockDiffraction(self,Radius, focalLength, x0, y0):
        DiffPattern = torch.angle(torch.exp(-1j*2*torch.tensor(torch.pi)*(self.X *x0 + self.Y *y0)/torch.tensor(self.wl)/focalLength))
        DiffPattern[:] *= torch.sqrt(self.X **2 + self.Y **2) < Radius
        self.field[:] = torch.abs(self.field)*torch.exp(1j*(torch.angle(self.field )+ DiffPattern)) 
        self.fourier[:] = self.F (self.field)

    def Axicon(self, kp = 300):
        phi = torch.arctan2(self.Y,self.X)
        self.field[:] *= torch.exp(1j*kp*torch.cos(phi)*self.X + 1j*kp*torch.sin(phi)*self.Y)
        self.fourier[:] = self.F(self.field)
    
    def Ring(self, innerRadius, outerRadius):
        self.field[:] = (torch.sqrt(self.X**2 + self.Y**2) > innerRadius) & (torch.sqrt(self.X**2 + self.Y**2) < outerRadius)
        self.fourier[:] = self.F(self.field)

    def CylindricalXLens(self, focalLength):
        self.field[:] *= torch.exp(-1j*self.k*(self.X**2)/2/focalLength)
        self.fourier[:] = self.F(self.field)

    def CylindricalYLens(self, focalLength):
        self.field[:] *= torch.exp(-1j*self.k*(self.Y**2)/2/focalLength)
        self.fourier[:] = self.F(self.field)

    def load_field(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.pixelNumber, self.pixelNumber), interpolation=cv2.INTER_CUBIC) 
        image = 255 - torch.from_numpy(image)
        self.field[:] = image/torch.max(image)
        self.fourier[:] = self.F(self.field)
    
    def get_intensity(self):
        return torch.abs(torch.Tensor.cpu(wf.field))

    def get_phase(self):
        return torch.angle(torch.Tensor.cpu(wf.field))
    
    def get_kspace(self):
        return torch.abs(torch.Tensor.cpu(torch.fft.fftshift(wf.fourier)))

if __name__ == "__main__":
    wf = Wavefront(pixelNumber= 2**12, Lfield = 10e-3, wl = 1064e-9, device = 'cuda')
    wf.Gauss(beamFwhm=3e-3)
    # wf.Vortex(Radius=3e-3)
    wf.Axicon(kp = 5000)
    # wf.ThinLens(focalLength=10e-3)
    # wf.propagate_angular_spec(distance=10e-3)

    plt.subplot(131)
    plt.imshow(wf.get_intensity())
    plt.subplot(132)
    plt.imshow(wf.get_kspace())
    plt.subplot(133)
    plt.imshow(wf.get_phase())
    plt.show()
    # start_time = time.time()
    # for i in range(10000):
    #     wf.F(wf.field)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('Elapsed time: ', elapsed_time)