import h5py
import numpy as np
import torch

def min_max_norm(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    return(x)
def datavalue(sp_size):
    f_data = h5py.File("sdss_galaxy_spec.hdf5", "r")

    # Fiber identification
    plateid = f_data['raw']['plateid'][()]
    fiberid = f_data['raw']['fiberid'][()]
    mjd = f_data['raw']['mjd'][()]

    # Position
    ra = f_data['raw']['ra'][()]
    dec = f_data['raw']['dec'][()]
    z = f_data['raw']['z'][()]
    zerr = f_data['raw']['zerr'][()]

    # Physical properties
    age = f_data['raw']['age'][()]
    metallicity = f_data['raw']['metallicity'][()]
    smass = f_data['raw']['smass'][()]

    # Select data
    condition1 = metallicity > 0.0
    condition2 = smass <= 1e13
    condition = np.logical_and(condition1, condition2)

    age = f_data['raw']['age'][condition]
    metallicity = f_data['raw']['metallicity'][condition]
    smass = f_data['raw']['smass'][condition]
    z = f_data['raw']['z'][condition]

    # Spectra
    wavelength = f_data['raw']['wavelength'][()]

    specs = f_data['raw']['spec'][condition]
    specerrs = f_data['raw']['specerr'][condition]
    print('Necessary Shape')
    print('spec:',specs.shape,'age',age.shape,'metallicity',metallicity.shape,'smass',smass.shape,'z',z.shape)

    condition = np.stack((min_max_norm(age), 
                          min_max_norm(metallicity), 
                          min_max_norm(smass),
                          min_max_norm(z)
                          ), axis=1)
    print('Shape of condition (shape):',condition.shape)

    shape  =specs.shape
    nbins = sp_size
    res = shape[1]//nbins
    specs_low_res = specs[:,:nbins * res].reshape(shape[0], nbins, res).mean(axis=2)
    wavelength_low_res = wavelength[:nbins * res].reshape(nbins, res).mean(axis=1)
    shape = specs_low_res.shape
    print('Shape of the low energy bins spectra',shape)
    return specs_low_res,condition

def prepareDataSet(specs_low_res,condition,train_len,val_len,device,batchsize,shuffle=True):
    def assignData(ndata_train,ndata_val,data):
        training_data = data[:ndata_train-ndata_val]
        val_data = data[ndata_train-ndata_val:ndata_train]
        test_data = data[ndata_train:]

        # convert to torch tensor
        training_data = torch.tensor(training_data, dtype=torch.float32).to(device)
        val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
        return training_data,val_data,test_data

    def DataLoader(X,Y,batchsize):
        dataset = torch.utils.data.TensorDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
        return train_loader
    # normalization
    normalized_data = specs_low_res - np.min(specs_low_res) + 1
    normalized_data = np.log10( normalized_data )
    min_val, max_val = np.min(normalized_data), np.max(normalized_data)
    normalized_data = (normalized_data - min_val) / (max_val - min_val)
    # divide into training / validation / test data
    ndata = len(specs_low_res)
    ndata_train = train_len
    ndata_val = val_len

    training_data,val_data,test_data= assignData(ndata_train,ndata_val,normalized_data)
    training_label,val_label,test_label=assignData(ndata_train,ndata_val,condition)
    

    print("training data size: ", training_data.size(),'label: ',training_label.size())
    print("validation data size: ", val_data.size(),'label: ',val_label.size())
    print("test data size: ", test_data.size(),'label: ',test_label.size())
<<<<<<< HEAD

    return DataLoader(training_data,training_label,batchsize)
=======
    return DataLoader(training_data,training_label)
>>>>>>> parent of 364e339 (Training and file)




