import h5py
import numpy as np
from IPython import embed
from matplotlib.pylab import plt


efit_name = [RXPT1, RXPT2, ZXPT1 ,ZXPT2 , Z0,R0 ,TRIBOT, TRITOP, KAPPA ,AMINOR, DRSEP]
power_params = [P_SOL,P_ldivL,P_ldivR,P_udivL,P_udivR,P_ldiv,P_udiv, P_core,P_axis,P_tot]



# ------------------------
# Model Definition
# ------------------------
def silu(x):
    """NumPy implementation of the SiLU (Swish) activation function."""
    return x / (1 + np.exp(-x))

def mlp(params, x):
    """implementation of the MLP forward pass."""
    for W, b in params[:-1]:
        x = np.dot(x, W) + b
        x = silu(x)
    W, b = params[-1]
    return np.dot(x, W) + b

def predict(params, Wlin, W0, x):
    """Full prediction function."""
    return W0 + np.dot(x, Wlin) + mlp(params, x)

# ------------------------
# Network Loading
# ------------------------
def load_network(filepath):
    """Loads the network parameters from an HDF5 file into NumPy arrays."""
    params = []
    with h5py.File(filepath, 'r') as f:
        # Load MLP parameters
        i = 0
        while f'mlp/layer_{i}/W' in f:
            W = f[f'mlp/layer_{i}/W'][:]
            b = f[f'mlp/layer_{i}/b'][:]
            params.append([W, b])
            i += 1

        # Load linear part parameters
        Wlin = f['linear/Wlin'][:]
        W0 = f['linear/W0'][:]

    print(f"Network loaded successfully from {filepath}")
    return params, Wlin, W0

# ------------------------
# Main Application Logic
# ------------------------
def apply_model(params, Wlin, W0, X, Y, batch_size=4096):
    """
    Applies the loaded network to input data X and Y to get predictions P_hat.
    Computes W_hat = network(X) and then P_hat = W_hat @ Y.
    """

    num_ch = Y.shape[1] # Inferred from problem, should be passed or stored if variable
    Dout = W0.size
    num_p = Dout // num_ch

   

    # Predict the weights W_hat for the batch
    W_hat_flat = predict(params, Wlin, W0, X)
    W_hat = W_hat_flat.reshape(-1, num_p, num_ch)

    # Apply the weights to Y to get the final prediction P_hat
    P_hat = np.einsum("nd,nkd->nk", Y, W_hat)


    return P_hat


# ------------------------
# Data Loading
# ------------------------
def load_data(file_path):
    """Loads and preprocesses data from the HDF5 file."""
    print(f"Loading data from {file_path}...")

    with h5py.File(file_path, "r") as f:
        channels = f["channels"][:]

        EFIT_values = np.array([f[k][:] for k in efit_name])
        power_values = np.array([f[k][:] / 1e6 for k in power_params]) # MW
        synthetics_brightness = f["synthetics_brightness+noise"][:] / 1e6 # MW/m^2
        missing_channels = [ch.strip() for ch in f["missing_channels"][:]]
    
    # --- Data Cleaning & Reshaping (as in original script) ---
    EFIT_values[-1, (EFIT_values[-1] > 0.3) | (EFIT_values[-1] < -0.3)] = 0
    EFIT_values[2, EFIT_values[0] > 1.8] = -1.15
    EFIT_values[0, EFIT_values[0] > 1.8] = 1.25
    EFIT_values[2, EFIT_values[0] < 0] = -1.15
    EFIT_values[0, EFIT_values[0] < 0] = 1.25
    EFIT_values[3, EFIT_values[1] > 1.8] = 1.2
    EFIT_values[1, EFIT_values[1] > 1.8] = 1.2
    EFIT_values[3, EFIT_values[1] < 0] = 1.2
    EFIT_values[1, EFIT_values[1] < 0] = 1.2

    valid_ch = ~np.in1d(channels, missing_channels)
    synthetics_brightness_valid = synthetics_brightness[:, valid_ch]
    nch = valid_ch.sum()

    Y = synthetics_brightness_valid.T.reshape(nch, -1, 1000).swapaxes(0, 1)
    P = power_values.reshape(len(power_params), -1, 1000).swapaxes(0, 1)
    X = EFIT_values.T

    # Ensure contiguous arrays for better performance
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    P = np.ascontiguousarray(P, dtype=np.float32)
    print("Data loading complete.")
    return X, Y, P




def get_region_mask(t, rvec, zvec, ATIME, PsinEmiss,BdMat,ZXPT1,ZXPT2,RXPT1,RXPT2,R0,Z0):
    
    R,Z = np.meshgrid(rvec, zvec)

    a_nearest = np.argmin(np.abs(ATIME[:-1]-t))

    
    div_low_side = (R - RXPT1[a_nearest]) * (Z0[a_nearest] - ZXPT1[a_nearest]) - (Z - ZXPT1[a_nearest]) * (R0[a_nearest] - RXPT1[a_nearest])
    div_up_side = (R - RXPT2[a_nearest]) * (Z0[a_nearest] - ZXPT2[a_nearest]) - (Z - ZXPT2[a_nearest]) * (R0[a_nearest] - RXPT2[a_nearest])
   
    mask = PsinEmiss  * 0 # FarSOL
    mask[PsinEmiss<1.1] = 5 # edge
    mask[PsinEmiss<0.9] = 6 # core
    mask[PsinEmiss<0.2] = 7 # axis
    mask[(Z < ZXPT1[a_nearest] + 0.2) & (div_low_side > 0) & (ZXPT1[a_nearest] > -2)&(PsinEmiss < 1.2) ] = 1 # ldivR
    mask[(Z < ZXPT1[a_nearest] + 0.2) & (div_low_side < 0) & (ZXPT1[a_nearest] > -2)&(PsinEmiss < 1.2) ] = 2 # ldivL
    mask[(Z > ZXPT2[a_nearest] - 0.2) & (div_up_side < 0) & (ZXPT2[a_nearest] > -2)&(PsinEmiss < 1.2) ] = 3 # udivR
    mask[(Z > ZXPT2[a_nearest] - 0.2) & (div_up_side > 0) & (ZXPT2[a_nearest] > -2)&(PsinEmiss < 1.2)] = 4 # udivL
    mask[ BdMat ] = -1
    
    return mask



def load_tomography(shot):

 
        
    import glob

    try:
        file = glob.glob(f"./Database/Emissivity_*_{shot}.npz")[0]
    except IndexError:
        return {}

        
    emiss = dict(np.load( file, allow_pickle=True))

    import MDSplus
    mdsserver = 'localhost'
    MDSconn = MDSplus.Connection(mdsserver)

    tree = 'EFITRT1'
    MDSconn.openTree(tree, shot)
    SSIMAG = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIMAG').data()
    valid = SSIMAG != 0
    SSIMAG = SSIMAG[valid]

    #PSIN = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIN').data()
    PSIRZ = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIRZ').data()[valid]
    GTIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:GTIME').data()[valid]/1e3
    SSIBRY = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIBRY').data()[valid]

    Rgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:R').data()
    Zgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:Z').data()
    ATIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:ATIME').data()[valid]/1e3


    AEQDSK_data = {}
    for ename in efit_name:
        AEQDSK_data[ename] = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:{ename}').data()[valid]
    


    PSIN = (PSIRZ - SSIMAG[:,None,None])/(SSIBRY-SSIMAG)[:,None,None]
    
        
    #convert form float16 and renormalise
    gres = np.single(emiss['gres']) * emiss['gres_norm']


    rvec = emiss['rvec']
    zvec = emiss['zvec']
    tvec = emiss['tvec']
    dr = rvec[1]-rvec[0]
    dz = zvec[1]-zvec[0]
    R,Z = np.meshgrid(rvec, zvec)

    PsinEmiss = np.zeros_like(gres)
     
    emiss_regions = {p:[] for p in power_params}
    from scipy.interpolate import RectBivariateSpline
    for it, t in enumerate(emiss['tvec']):
        g_nearest = np.argmin(np.abs(GTIME-t))
        PsinEmiss = RectBivariateSpline(Zgrid,Rgrid,PSIN[g_nearest])(zvec, rvec)


        mask = get_region_mask(t, rvec, zvec, ATIME, PsinEmiss , emiss['BdMat'], 
                                AEQDSK_data['ZXPT1'],AEQDSK_data['ZXPT2'],
                                AEQDSK_data['RXPT1'],AEQDSK_data['RXPT2'],
                                AEQDSK_data['R0'],AEQDSK_data['Z0'])
        
        power = np.single(gres[:,:,it] * R  * 2 * np.pi * dr * dz) #W per cell
 
        emiss_regions['P_ldivL'].append(np.sum(power[mask == 2],0))
        emiss_regions['P_ldivR'].append( np.sum(power[mask == 1],0))
        emiss_regions['P_ldiv'].append(np.sum(power[(mask == 1)|(mask==2) ],0))
        
        emiss_regions['P_udivL'].append( np.sum(power[mask == 4],0))
        emiss_regions['P_udivR'].append(np.sum(power[mask == 3],0))
        emiss_regions['P_udiv'].append(np.sum(power[(mask == 3)|(mask==4)],0))

        emiss_regions['P_SOL'].append(np.sum(power[(mask == 0)|(mask==1)],0))

        emiss_regions['P_axis'].append(np.sum(power[mask == 7],0))
        emiss_regions['P_core'].append(np.sum(power[(mask == 7)|(mask==6)],0))
        emiss_regions['P_tot'].append(np.sum(power[mask>=0],0))
  

            
    
    emiss_regions = {p: np.array(v) for p,v in emiss_regions.items()}
    emiss_regions['tvec'] = emiss['tvec']
    
    
    return emiss_regions


def load_mds_data(shot, EFIT = 'EFITRT1', realtime_bolo=True):

    if realtime_bolo:
        print('Load realtime BOLO data')
    else:
        print('Load standart BOLO data')
        
    import MDSplus
    mdsserver = 'atlas.gat.com'
    MDSconn = MDSplus.Connection(mdsserver)

   
    MDSconn.openTree(EFIT, shot)
    AEQDSK_data = {}
    for ename in efit_name + ['ATIME']:
        AEQDSK_data[ename] = MDSconn.get(f'\\{EFIT}::TOP.RESULTS.AEQDSK:{ename}').data()
    
        
    etendue = { 'U':  [3.0206e4,2.9034e4,2.8066e4,2.7273e4,2.6635e4,4.0340e4,\
            3.9855e4,3.9488e4,3.9235e4,3.9091e4,3.9055e4,3.9126e4,\
            0.7972e4,0.8170e4,0.8498e4,0.7549e4,0.7129e4,0.6854e4,\
            1.1162e4,1.1070e4,1.1081e4,1.1196e4,1.1419e4,1.1761e4],
                'L': [2.9321e4,2.8825e4,2.8449e4,2.8187e4,2.8033e4,0.7058e4,\
            0.7140e4,0.7334e4,0.7657e4,0.8136e4,0.8819e4,0.7112e4,\
            0.6654e4,0.6330e4,0.6123e4,2.9621e4,2.9485e4,2.9431e4,\
            2.9458e4,2.9565e4,2.9756e4,3.0032e4,3.0397e4,0.6406e4]}
    
    #channels not availible in realtme 
    missing_channels =  ['U01', 'L11', 'L19', 'L20']
    if not realtime_bolo:
        TDIcall = "_x=\\BOLOM::TOP.PRAD_01.POWER:"
        MDSconn.openTree('BOLOM', shot)


    from scipy.signal import lfilter

    def lowpass_filter(x, alpha):
        b = [alpha]
        a = [1, -(1 - alpha)]
        return lfilter(b, a, x)
    
    bolo_brightness = []
    #load realtime bolometer power 
    for fan in ['U', 'L']:
        for ich in range(24):
            ch = f'{fan}{ich+1:02}'
       
            if ch in missing_channels:
                continue
            
            #reatime hadata by itself has ~50ms delay
            if realtime_bolo:
                data = MDSconn.get(f'_x=PTDATA2("DGSDPWR{ch}", {shot})').data()
                #this adds a small delay, but negligible compared to the existing delay in the data
                data = lowpass_filter(data, alpha=0.01)
            else:
                data = MDSconn.get(TDIcall+f'BOL_{ch}_P').data() #W
            if len(data) <= 1:
                raise Exception(f'No data for channel {ch}')
         
            data *= etendue[fan][ich] * 1e4 #W/m^2
            bolo_brightness.append(data) 
            
            
            
            
    tvec = MDSconn.get('dim_of(_x)').data()  #ms
    bolo_brightness = np.array(bolo_brightness)
    
    
    MDSconn.openTree('BOLOM', shot)


    power_params = ['P_SOL','P_ldivL','P_ldivR','P_udivL','P_udivR','P_ldiv','P_udiv', 'P_core','P_axis','P_tot']

    legacy_power = {}
    legacy_power['P_tot'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.PRAD.PRAD_TOT').data() #W
    legacy_power['P_ldiv'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.PRAD.PRAD_DIVL').data() #W
    legacy_power['P_udiv'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.PRAD.PRAD_DIVU').data() #W
    legacy_power['time'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.TIME').data() #ms
  
    legacy_power['P_core'] = MDSconn.get('\\BOLOM::TOP.BOLFIT01.ONED.POWER_CORE').data() #W
    legacy_power['P_SOL'] = MDSconn.get('_x=\\BOLOM::TOP.BOLFIT01.ONED.POWER_SOL').data() #W
    tvec_bolo = MDSconn.get('dim_of(_x)').data()  #ms
    
    legacy_power['P_core'] = np.interp(legacy_power['time'], tvec_bolo, legacy_power['P_core'])
    legacy_power['P_SOL'] = np.interp(legacy_power['time'], tvec_bolo, legacy_power['P_SOL'])
    
  

    #reinterpolate on realtime bolo timegrid 
    nearest = AEQDSK_data['ATIME'][:-1].searchsorted(tvec)
    inputs = np.array([AEQDSK_data[ename][nearest] for ename in efit_name])
    
    # --- Data Cleaning & Reshaping (as in original script) ---
    inputs[-1, (inputs[-1] > 0.3) | (inputs[-1] < -0.3)] = 0
    inputs[2, inputs[0] > 1.8] = -1.15
    inputs[0, inputs[0] > 1.8] = 1.25
    inputs[2, inputs[0] < 0] = -1.15
    inputs[0, inputs[0] < 0] = 1.25
    inputs[3, inputs[1] > 1.8] = 1.2
    inputs[1, inputs[1] > 1.8] = 1.2
    inputs[3, inputs[1] < 0] = 1.2
    inputs[1, inputs[1] < 0] = 1.2
  
    
    return tvec, inputs.T, bolo_brightness.T, legacy_power
        
 
# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    # --- Load the saved network ---
    network_file = 'trained_network.h5'
    mlp_params, Wlin, W0 = load_network(network_file)
    
    import sys
    real_time = True
    if len(sys.argv) > 2:
        shot = int(sys.argv[1])
        real_time = False
    elif len(sys.argv) > 1:
        shot = int(sys.argv[1])
    else:
        shot = 203401
    
        
    power_tomo = load_tomography(shot)
    
    
    tvec, X, Y, legacy_power = load_mds_data(shot, realtime_bolo=real_time)
    

    # --- Apply the network to the new data ---
    P_predicted = apply_model(mlp_params, Wlin, W0, X, Y)
    

    f,ax = plt.subplots(2,5, sharex=True, sharey=True, figsize=(10,8))
    ax = np.ravel(ax)
    for i, p in enumerate(power_params):
        ax[i].set_title(p)
        ax[i].plot(tvec/1e3,  P_predicted[:,i],'b-')
        ax[i].plot(power_tomo.get('tvec',0),  power_tomo.get(p,0),'r--')

      
        if p in legacy_power:
            ax[i].plot(legacy_power['time']/1e3, legacy_power[p],':')
            
            
        ax[i].axhline(0)
    ax[0].set_xlim(0, 7)
    ax[0].set_ylim(0, np.median(P_predicted[P_predicted[:,-1] > np.median(P_predicted[:,-1]),-1]) * 2)
    plt.tight_layout()
    f.savefig(f'bolo_{shot}')

    
    plt.show()
    

 
