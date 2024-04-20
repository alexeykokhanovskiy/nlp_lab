import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time
import h5py
import plotly.graph_objects as go

class Wavefront():
    def __init__(self, pixelNumber = 2**10, Lfield = 5e-3, wl = 1064e-9, device = 'cuda'):
        # Field Grid parameters
        if device == "cuda":
            if torch.cuda.is_available(): 
                self.device = 'cuda'
            else:
                self.device = 'cpu' 
                print("GPU is unavailable")
        else:
            self.device = device
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
        self.field  = self.field.to(self.device, dtype = torch.complex128)
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
        prop_factor = torch.exp(1j*distance*self.kz)
        prop_factor = prop_factor.to(self.device, dtype=torch.complex128)
        self.fourier[:] *= prop_factor 
        self.field[:] = self.iF(self.fourier)  

    def ThinLens(self, focalLength):
        transCoeff = torch.exp(-1j*self.k*(self.X**2+self.Y**2)/2/focalLength)
        self.field[:] = self.field*transCoeff
        self.fourier[:] = self.F(self.field)

    def Vortex(self,Radius):
        self.field[:] *= torch.exp(1j*(torch.arctan2(self.Y , self.X )))*(torch.sqrt(self.X **2 + self.Y **2) < Radius)
        self.fourier[:] = self.F (self.field )

    def Gauss(self, beamFwhm):
        self.field[:] = torch.exp(-2*torch.log(torch.tensor(2))*((self.X/beamFwhm)**2 + (self.Y/beamFwhm)**2)) 
        self.fourier[:] = self.F (self.field ) 
        return torch.exp(-2*torch.log(torch.tensor(2))*((self.X/beamFwhm)**2 + (self.Y/beamFwhm)**2)) 
    
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

    def CylindricalLens(self, focalLength, angle):
        alpha = torch.tensor(torch.pi*angle/180)
        self.field[:] *= torch.exp(-1j*self.k*((self.X*torch.cos(alpha)+ self.Y*torch.sin(alpha))**2)/2/focalLength)
        self.fourier[:] = self.F(self.field)

    def load_field(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.pixelNumber, self.pixelNumber), interpolation=cv2.INTER_CUBIC) 
        image = 255 - torch.from_numpy(image)
        self.field[:] = image/torch.max(image)
        self.fourier[:] = self.F(self.field)
    
    def get_intensity(self):
        return torch.abs(torch.Tensor.cpu(self.field))

    def get_phase(self):
        return torch.angle(torch.Tensor.cpu(self.field))
    
    def get_kspace(self):
        return torch.abs(torch.Tensor.cpu(torch.fft.fftshift(self.fourier)))
    
class WavefrontStack():

    def __init__(self, wf, to_device = False):
        if to_device:
            self.stack = torch.empty([wf.field.shape[0], wf.field.shape[1], 1], dtype=torch.complex128)
            self.stack = self.stack.to(wf.device)
        else:
            self.stack = torch.empty([wf.field.shape[0], wf.field.shape[1], 1], dtype=torch.complex128)

    def insert_in_device(self, wf: Wavefront):
        self.stack = torch.cat((self.stack, wf.field.unsqueeze(2)), dim=-1)

    def insert(self, wf: Wavefront):
        if wf.device == 'cuda':
            self.stack = torch.cat((self.stack, torch.Tensor.cpu(wf.field.unsqueeze(2))), dim=-1)
        else:
            self.stack = torch.cat((self.stack, wf.field.unsqueeze(2)), dim=-1)
    
    def save(self, path):
        hf = h5py.File(path, 'w')
        hf.create_dataset('wavefrontstack', data = torch.Tensor.cpu(self.stack).numpy())
        hf.close()
    
    def load(self,path):
        hf = h5py.File(path, 'r')
        self.stack = hf.get('wavefrontstack')

    def get_shape(self):
        return self.stack.shape

    def resize(self, size):
        wfStack_resized = WavefrontStack()
        wfStack_resized.stack = np.zeros((size,size, self.stack.shape[2]))


    def plot_slices_3d(self, I):
        I = np.abs(wfStack_compressed)
        max_intensity = np.max(I)
        print(max_intensity)
        # print(wfStack.stack[:,:,1].shape)
        # fig = go.Figure(data=[go.Surface(z=np.abs(wfStack.stack))])
        nb_frames = 60
        r, c =  I[:,:,0].shape
        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            z=k * np.ones((r, c)),
            surfacecolor=(np.abs(I[:,:,60-k])),
            cmin=0, cmax=5
            ),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)])
        k = 0
    # Add data to be displayed before animation starts
        fig.add_trace(go.Surface(
            z=6.7 * np.ones((r, c)),
            surfacecolor=(np.abs(I[:,:, 60-k])),
            colorscale='jet',
            cmin=0, cmax=5,
            colorbar=dict(thickness=20, ticklen=4)
            ))

        def frame_args(duration):
            return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

        sliders = [
                    {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], frame_args(0)],
                                "label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig.frames)
                        ],
                    }
                ]    

        fig.update_layout(
            title='Slices in volumetric data',
            width=800,
            height=800,
            scene=dict(
                        zaxis=dict(range=[0, 60], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )


if __name__ == "__main__":

    # nb_frames = 31
    wf = Wavefront(pixelNumber= 2**10, Lfield = 10e-3, wl = 1064e-9, device = 'cpu')
    wfStack = WavefrontStack(wf, to_device=True)
    wf.Gauss(beamFwhm=1e-3)
    wf.Vortex(Radius=1e-3)
    wf.CylindricalLens(focalLength=150e-3, angle = 45)
    step = 5e-3
    nb_frames = 60
    z = np.linspace(0, 300e-3, nb_frames)
    # step = z[1] - z[0]
    for item in z:
        wf.propagate_angular_spec(step)
        wfStack.insert_in_device(wf)

    wfStack.save('test.h5')
    wfStack.load('test.h5')
    
    wfStack_compressed = np.zeros((256,256, wfStack.stack.shape[2]))
    for i in range(0, wfStack.stack.shape[2]):
        wfStack_compressed[:,:,i] = image = cv2.resize(np.abs(wfStack.stack[:,:,i]), (256, 256), interpolation=cv2.INTER_LINEAR) 
    
    I = np.abs(wfStack_compressed)
    max_intensity = np.max(I)
    print(max_intensity)
    # print(wfStack.stack[:,:,1].shape)
    # fig = go.Figure(data=[go.Surface(z=np.abs(wfStack.stack))])
    nb_frames = 60
    r, c =  I[:,:,0].shape
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=k * np.ones((r, c)),
        surfacecolor=(np.abs(I[:,:,60-k])),
        cmin=0, cmax=5
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])
    k = 0
# Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=6.7 * np.ones((r, c)),
        surfacecolor=(np.abs(I[:,:, 60-k])),
        colorscale='jet',
        cmin=0, cmax=5,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]    

    fig.update_layout(
         title='Slices in volumetric data',
         width=800,
         height=800,
         scene=dict(
                    zaxis=dict(range=[0, 60], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
)

    fig.show()
#     # I = np.abs(torch.Tensor.cpu(wfStack.stack).numpy())
    # max_intensity = np.max(I)
    # Z = np.arange(0, 31, 1)
    # X, Y, Z  = np.meshgrid(wf.scale, wf.scale, Z)
    # fig = go.Figure(data=go.Isosurface(
    #         x=X.flatten(),
    #         y=Y.flatten(),
    #         z=Z.flatten(),
    #         value=I.flatten(),
    #         opacity=0.6,
    #         isomin=0.1 * max_intensity,
    #         isomax=0.9 * max_intensity,
    #         surface_count=10,
    #         colorbar_nticks=10, # colorbar ticks correspond to isosurface values
    #         # caps=dict(x_show=False, y_show=False)
    #     ))    
    # fig.update_layout(title = 'title')
    # fig.show()


    # stack = WavefrontStack(wf)
    # # stack.save('test.h5')
    # stack.load('test.h5')
    # print(stack.get_shape())
    # stack = torch.empty([wf.field.shape[0], wf.field.shape[1], 1])
    # print(stack.shape)
    # print(wf.field.unsqueeze(2).shape)
    # stack = torch.cat((stack, wf.field.unsqueeze(2)), dim=-1)
    # stack = torch.cat((stack, wf.field.unsqueeze(2)), dim=-1)
    
    # print(wf.field.unsqueeze(2).shape)    
    # print(wf.field)
    # print(stack.shape)
    # stack = torch.stack((stack, wf.field), -1)

    # print(stack.shape)
    # stack = torch.stack((stack, stack), 1)
    

    # start_time = time.time()
    # for i in range(10000):
    #     wf.F(wf.field)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('Elapsed time: ', elapsed_time)