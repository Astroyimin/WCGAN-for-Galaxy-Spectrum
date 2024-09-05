import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import  os
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True   
mpl.rcParams['figure.dpi'] = 300


def renormdata(spec,max_val,min_val):
    spec = spec * ( (max_val - min_val)) + min_val
    spec = np.power(10, spec)
    spec = spec+np.min(spec)-1
    return spec

def get_spec_plot(x,y,title):
    wavelength = x 
    min_val, max_val = np.min(y), np.max(y)
    spec = renormdata(y,max_val,min_val)
    plt.figure(figsize=(5,4))
    plt.plot(wavelength, spec)
    plt.xlabel('wavelength [angstrom]')
    plt.ylabel('spectrum')
    plt.xlabel('wavelength [angstrom]')
    plt.ylabel('spectrum')
    bath = 'Image'
    if not os.path.exists(bath):
        os.mkdir(bath)
    path = os.path.join(bath,title+'.png')
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()
    image = plt.imread(path)

    return image

