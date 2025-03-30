import numpy as np
import BayesPower as BP
import pickle, json, os, sys
from enterprise_extensions.model_utils import get_tspan

# pip install fastshermanmorrison-pulsar

fbins = 5
pidx = int(sys.argv[1])

datadir = '/home/koonima/FAST/Data/Pickle/'
with open(datadir + f'v1p1_de440_pint_bipm2019.pkl', 'rb') as fin:
    psrs = pickle.load(fin)[pidx]

m = BP.BayesPowerSingle(psr = psrs,
            Tspan = get_tspan([psrs]),
            select = 'backend',
            white_vary = True,
            inc_ecorr = True,
            tnequad=False,
            ecorr_type = 'kernel',
            noise_dict = None,
            tm_marg = False,
            freq_bins = fbins,
            log10rhomin=-9.0,
            log10rhomax=-4.5,)

m.sample(niter=int(4e5),
        resume=False,
        progress_bar=True,
        savepath = f'./Chain/{fbins}_FS_SP')