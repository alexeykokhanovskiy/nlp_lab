import LightPipes
from Wavefront import Wavefront
import matplotlib.pyplot as plt
import torch
import time

def import_light_pipe(Field: LightPipes.Field, device = 'cuda'):
    wf = Wavefront(pixelNumber=Field.N, Lfield = Field.grid_size, wl = F.wavelength, device = device)
    wf.field[:] = torch.sqrt(torch.from_numpy(LightPipes.Intensity(Field)))*torch.exp(1j*torch.from_numpy(LightPipes.Phase(Field)))
    wf.fourier[:] = wf.F(wf.field)
    return wf

if __name__ == "__main__":
  
    labda=632.8e-9
    size=8e-3
    N=2**12


    R1=600e-3
    R2=437e-3
    np_=0
    nl_=4
    mh=0
    nh=4


    df2=12e-3

    f1=20e-3
    f2=160e-3 +df2 #160*mm
    f3=19e-3
    f4=19e-3
    f5=50e-3

    dz=-30e-3 # tune position lens f2
    d1=525e-3
    d2=306e-3
    d3=225e-3 + dz #225*mm
    d4=176e-3 - dz #176*mm
    d5=27e-3 #27*mm
    fM2=-R2/(1.5-1) #lensmakers formula with refractive index = 1.5, focal length outcoupler

    L=d1+d2
    g1=1-L/R1
    g2=1-L/R2   
    w0=torch.sqrt(torch.tensor(labda*L/torch.pi))
    w0*=(g1*g2*(1-g1*g2))**0.25
    w0/=(g1+g2-2*g2)**0.5
    F=LightPipes.Begin(size,labda,N)
    F=LightPipes.GaussLaguerre(F, w0, p=np_, l=nl_, A=1.0, ecs=0)
    wf = import_light_pipe(F, "")
    start_time = time.time()
    wf.propagate_angular_spec(d2)
    wf.ThinLens(focalLength=fM2)
    wf.propagate_angular_spec(d3)
    wf.ThinLens(f2)
    wf.propagate_angular_spec(d4)
    wf.CylindricalLens(f3, angle = 45)
    wf.propagate_angular_spec(d5)
    wf.CylindricalLens(f4, angle = 45)
    wf.propagate_angular_spec(27e-3)
    wf.propagate_angular_spec(25e-3) # propagate to have sufficient large beam size
    plt.imshow(wf.get_intensity())
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time: ', elapsed_time)
   
