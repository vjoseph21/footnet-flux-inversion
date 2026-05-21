import sys
import yaml
import pickle
import datetime, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely import Polygon, Point
import netCDF4 as nc
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.img_tiles import GoogleTiles
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix
from scipy.sparse import save_npz, load_npz
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import inv
from pprint import pprint

from get_TROPOMI_obs import getTROPOMI
from emissions import getEmissions

args = sys.argv[1:]

class TROPOMI_config():
    def __init__(self, cfs):
        self.tropomi_filepath = cfs["tropomi_filepath"]
        self.tropomi_train_file = cfs["tropomi_train_file"]
        self.tropomi_test_file = cfs["tropomi_test_file"]
        self.bkg_file = cfs["tropomi_bkg_file"]
        self.ems_buffer_days = cfs["ems_buffer_days"]
        
        self.start_time = datetime.datetime.strptime(cfs["start_time"], "%Y%m%d%H")
        self.end_time = datetime.datetime.strptime(cfs["end_time"], "%Y%m%d%H")

        self.week_start_date = self.start_time - datetime.timedelta(days=self.start_time.weekday() + 1 if self.start_time.weekday() < 6 else 0) - datetime.timedelta(hours=self.start_time.hour)
        self.inv_start_time = self.week_start_date - datetime.timedelta(days=self.ems_buffer_days)
        
        self.week_end_date = self.end_time - datetime.timedelta(days=self.end_time.weekday() + 1 if self.end_time.weekday() < 6 else 0) - datetime.timedelta(hours=self.end_time.hour)
        self.inv_end_time = self.week_end_date + datetime.timedelta(days=self.ems_buffer_days) # datetime.datetime(2021, 1, 2, 23)
    
        self.xres = cfs["xres"]
        self.yres = cfs["yres"]
        self.clon = cfs["clon"]
        self.clat = cfs["clat"]

        data = np.load(cfs["lat_lon_file"])
        self.lons = data["lon"]
        self.lats = data["lat"]
    
        self.upwind_degree_margin = cfs["upwind_degree_margin"]
        self.upwind_date_margin = cfs["upwind_date_margin"]
        self.obs_indices_margin = cfs["obs_indices_margin"]
        self.met_temp_resolution_background = cfs["met_temp_resolution_background"] # hours wind data for computing background 

        self.hr3latlon_mapping = cfs["HRRR_LAT_LON_MAPPING"]
        self.hr3lon_full = np.load(self.hr3latlon_mapping)['lon']
        self.hr3lat_full = np.load(self.hr3latlon_mapping)['lat']
        self.hr3lon_full = (self.hr3lon_full+180)%360-180  # convert from 0~360 to -180~180
        self.HRRR_DIR = cfs["HRRR_DIR"]
        self.trimsize = cfs["trimsize"]
    
        self.model_path = cfs["model_path"]
        self.footprint_path = cfs["footprint_path"]
        self.maximum_domain_trajectory = cfs["maximum_domain_trajectory"]

        self.inventory_type = cfs["inventory_type"]
        self.inventory_path = cfs["inventory_path"]
        self.ems_uncert = cfs["ems_uncert"]
        self.ems_scaling_factor = cfs["ems_scaling_factor"]

        self.output_path = cfs["output_path"]

class Arguments():
    def __init__(self):
        self.config = str(args[0])
        self.start_time = str(args[1])
        self.end_time = str(args[2])

def get_stored_df(path, start_time, end_time):
    obs_df = pd.read_csv(path)
    obs_df['delta_time'] = obs_df['delta_time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
    obs_df['actual_time'] = obs_df['actual_time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
    obs_df['time'] = obs_df['time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    obs_df = obs_df[(obs_df["time"] >= start_time) & (obs_df["time"] <= end_time)]
    obs_df = obs_df.sort_values(["time", "lon", "lat"])
    obs_df["bkg_ref"] = obs_df["bkg_ref"].apply(lambda x: eval(x))
    return obs_df.reset_index(drop=True)

def get_obs_dict(obs_df):
    obs_dict = {}
    temp = obs_df.to_dict(orient='records')
    for idx in range(obs_df.shape[0]):
        obs_dict[idx] = temp[idx]
    return obs_dict

def load_background_data(config):
    with open(config.bkg_file, "rb") as f:
        bkg_data = pickle.load(f)
    bkg_dict, bkg_date_range = bkg_data["background_dict"], bkg_data["background_date_range"]
    bkg_date_range = list(bkg_date_range)
    return bkg_dict, bkg_date_range

def fill_missing_values(bkg, date_range):
    bkg[bkg==0] = None
    bkg = list(pd.Series(bkg, index=date_range).interpolate(method='linear').ffill().bfill())
    return bkg

def get_bkg_prior_error():    
    xp_bkg_dict = {}
    xp_bkg_error_dict = {}
    
    xp_bkg = np.zeros((len(bkg_dict.keys()), len(xp_bkg_date_range)))
    xp_bkg_error = np.zeros((len(bkg_dict.keys()), len(xp_bkg_date_range)))
    error_mask = np.ones((len(bkg_dict.keys()), len(xp_bkg_date_range)))
    
    for idx, val in train_obs_dict.items():
        tstamp = val["time"] - datetime.timedelta(hours=val["time"].hour)
        t_index = xp_bkg_date_range.index(tstamp)
        if t_index not in xp_bkg_dict:
            xp_bkg_dict[t_index] = {}
            xp_bkg_error_dict[t_index] = {}
        for octant in val["bkg_ref"]:
            octant_idx, octant_dist, octant_time, octant_bkg, octant_bkg_error = octant
            xp_bkg_dict[t_index][octant_idx-1] = xp_bkg_dict[t_index].get(octant_idx-1, []) + [octant_bkg]
            xp_bkg_error_dict[t_index][octant_idx-1] = xp_bkg_error_dict[t_index].get(octant_idx-1, []) + [octant_bkg_error]
    
    for idx, val in train_obs_dict.items():
        tstamp = val["time"] - datetime.timedelta(hours=val["time"].hour)
        t_index = xp_bkg_date_range.index(tstamp)
        for octant in val["bkg_ref"]:
            octant_idx, octant_dist, octant_time, octant_bkg, octant_bkg_error = octant
            xp_bkg[octant_idx-1, t_index] = np.mean(xp_bkg_dict[t_index][octant_idx-1])
            xp_bkg_error[octant_idx-1, t_index] = np.mean(xp_bkg_error_dict[t_index][octant_idx-1])
    
    error_mask = np.ones((len(bkg_dict.keys()), len(xp_bkg_date_range)))
    error_mask[xp_bkg_error==0] = 2
    xp_bkg[xp_bkg==0] = None
    xp_bkg_error[xp_bkg_error==0] = None
    for idx in range(xp_bkg.shape[0]):
        xp_bkg[idx, :] = fill_missing_values(xp_bkg[idx, :], xp_bkg_date_range)
        xp_bkg_error[idx, :] = fill_missing_values(xp_bkg_error[idx, :], xp_bkg_date_range)
    
    xp_bkg_error_adjusted = xp_bkg_error * error_mask
    xp_bkg = xp_bkg.reshape(-1, 1, order="F")
    return xp_bkg, xp_bkg_error_adjusted

def fill_H(config, dates, obs_dict):
    m = config.lons.shape[0] * config.lats.shape[0]
    n_obs = len(obs_dict)

    rows = []
    cols = []
    data = []

    for idx, value in tqdm(obs_dict.items()):
        date = value["time"]

        week_start_date = (
            date - datetime.timedelta(days=date.weekday() + 1 if date.weekday() < 6 else 0)
            - datetime.timedelta(hours=date.hour)
        )

        m_index = dates.get_loc(week_start_date)

        file = (
            f"{config.footprint_path}/{date.year}/{date.month}/"
            f"footnet_footprint_TROPOMI_GP_{date:%Y%m%d%H}_{value['lat']}_{value['lon']}.nc"
        )

        # load footprint
        with nc.Dataset(file) as f:
            foot = np.array(f["foot"]).reshape(-1, order="F").astype(np.float32)

        # get nonzero indices
        nz = foot.nonzero()[0]

        if nz.size > 0:
            rows.extend([idx] * nz.size)
            cols.extend(nz + m_index * m)
            data.extend(foot[nz])

    # build CSC
    H_old = csc_matrix(
        (data, (rows, cols)),
        shape=(n_obs, dates.shape[0] * m),
        dtype=np.float32
    )

    return H_old * 1000 # converting from ppm to ppb

def compute_H_background(obs_dict, config):
    xp_bkg_date_range = list(pd.date_range(start=config.inv_start_time, end=config.inv_end_time, freq="1D"))
    nObs = len(obs_dict)
    H_b_train = np.zeros((nObs, xp_bkg.shape[0]))
    
    for idx, val in obs_dict.items():
        tstamp = val["time"] - datetime.timedelta(hours=val["time"].hour)
        t_index = xp_bkg_date_range.index(tstamp)
        norm_dist = val["norm_dist"]
        for octant in val["bkg_ref"]:
            octant_idx, octant_dist, octant_time, octant_bkg, octant_bkg_error = octant
            H_b_train[idx, t_index*8+(octant_idx-1)] = 1/octant_dist/norm_dist 
    return H_b_train

def get_len(coords_1, coords_2):
    """
        Compute the length between two coordinates

        Arguments:
            coords_1: <list>
            coords_2: <list>

        returns:
            <float>
    """
    lat1 = coords_1[0]*np.pi/180
    lon1 = coords_1[1]*np.pi/180
    lat2 = coords_2[0]*np.pi/180
    lon2 = coords_2[1]*np.pi/180
    R = 6371e3
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # c = 2 ⋅ atan2( √a, √(1−a) )
    # d = R ⋅ c
    a = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon1-lon2)/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c/1000 #km


def inner_loop(i, nObs, obs_dict):
    tau_time = 1 # days
    tau_space = 25 # km
    res = [(i, i, 1)]
    time_val_i = obs_dict[i]["time"]
    coord_i = (obs_dict[i]["lat"], obs_dict[i]["lon"])
    for j in range(i+1, nObs):
        time_val_j = obs_dict[j]["time"]
        coord_j = (obs_dict[j]["lat"], obs_dict[j]["lon"])
        time_difference = np.abs((time_val_i - time_val_j).days)
        dist = np.abs(get_len(coord_i, coord_j))
        time_decay = np.exp(-time_difference/tau_time)
        dist_decay = np.exp(-dist/tau_space)
        sig_val = time_decay * dist_decay
        res.append((i, j, sig_val))
        res.append((j, i, sig_val))
    return res

def fill_R(obs_dict):
    nObs = len(obs_dict)
    So = np.zeros((nObs, nObs), dtype=np.float32)
    OUTPUT = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(delayed(inner_loop)(i, nObs, obs_dict) for i in range(nObs))
    for entry in tqdm(OUTPUT):
        for row, col, val in entry:
            So[row][col] = val
    So = csc_matrix(So)
    return So

def compute_D_E(Sa_xy, Sa_t, emission, ems_uncert):
    sigma_t = np.average(emission.emissions[np.nonzero(emission.emissions)])*ems_uncert
    D = csc_matrix(Sa_t_mod)*sigma_t
    sigma_ems = csc_matrix(emission.emissions.reshape((m, 1), order="F"))*ems_uncert
    E = Sa_xy.multiply(sigma_ems)
    return D, E

def compute_HQblock_thread(i, H, D, E, n, p, r, t):
    H_sum = None
    D_col = D[:, i].tocoo()

    for j, Dij in zip(D_col.row, D_col.data):
        H_block = H[:, j*r:(j+1)*r]
        scaled = H_block.multiply(Dij)
        H_sum = scaled if H_sum is None else (H_sum + scaled)

    if H_sum is None:
        return csc_matrix((n, t), dtype=np.float32)
    block = H_sum.toarray() @ E
    return csc_matrix(block)

def HQ_sparse_parallel(H, D, E, n_jobs=-1):
    E = E.toarray()
    n = H.shape[0]
    p, q = D.shape
    r, t = E.shape
    
    # THREADING backend → no additional memory copies
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(compute_HQblock_thread)(i, H, D, E, n, p, r, t)
        for i in tqdm(range(q))
    )

    print("Combining HQ blocks")
    HQ = lil_matrix((n, q*t), dtype=np.float32)
    for i, block in tqdm(enumerate(results)):
        HQ[:, i*t:(i+1)*t] = block

    return HQ.tocsc()

def HQHT_sparse(HQ_INDIRECT, H, D, E):
    """
    Computes HQHT = H * Q * H^T, using Q = kron(D, E)
    All arguments (HQ_INDIRECT, H, D, E) must be sparse.
    
    HQ_INDIRECT: sparse (n × q*t)   = H * kron(D, E)
    H          : sparse (n × p*r)
    D          : sparse (p × q)
    E          : sparse (r × t)

    Returns:
    HQHT: sparse (n × n)
    """

    n = H.shape[0]
    p, q = D.shape
    r, t = E.shape

    # Output HQHT as sparse (LIL for incremental construction)
    HQHT = lil_matrix((n, n), dtype=np.float32)

    counter = t

    for i in tqdm(range(q)):
        # block cols in HQ_INDIRECT
        col_start = i * t
        col_end   = (i + 1) * t

        # block cols in H
        row_start = i * r
        row_end   = (i + 1) * r

        # Extract sparse blocks
        HQ_block = HQ_INDIRECT[:, col_start:col_end]   # (n × t)
        H_block  = H[:, row_start:row_end]             # (n × r)

        # Sparse block multiplication
        # result is (n × n) sparse
        HQ_block = HQ_block.toarray()
        H_block = H_block.toarray()
        contrib = HQ_block @ H_block.T
        contrib = csc_matrix(contrib)
        # Add to HQHT
        HQHT += contrib

    return HQHT.tocsc()

def get_Sa_t(Xp, tau_week, tau_year, correlation_type="mod"):
    """
    Build temporal prior error covariance with weekly timestep and yearly periodic correlation.
    
    Args:
        Xp: prior emissions vector (1D)
        tau_week: correlation length (in weeks) for short-term temporal decay
        tau_year: correlation length (in weeks) for seasonal periodicity
        dates: list/array of datetime objects
        m: number of grid cells per timestep
        ems_uncert: fractional prior uncertainty
        correlation_type: "mod" for modular periodic, "exp" for exponential, "sin" for sine-based
    """
    nEms = int(Xp.shape[0] / m)
    Sa_t = np.zeros((nEms, nEms))
    
    for i in tqdm(range(nEms)):
        sigmai = 1
        for j in range(i, nEms):
            sigmaj = 1
            weeks_apart = abs(j - i)
            
            # Short-term correlation (week-to-week)
            temp_weeks = np.exp(-weeks_apart / tau_week)
            
            # Yearly correlation (periodic)
            if correlation_type == "mod":
                # Repeat correlation every 52 weeks (1 year)
                weeks_apart_mod = 26 - abs(26 - np.mod(weeks_apart, 52))
                temp_years = np.exp(-weeks_apart_mod / tau_year)
            elif correlation_type == "sin":
                years_apart = (dates[j] - dates[i]).days / 365
                temp_years = np.exp(-2 * (np.sin(np.pi * years_apart))**2 / tau_year**2)
            else:
                years_apart = (dates[j] - dates[i]).days / 365
                temp_years = np.exp(-abs(years_apart) / tau_year)
            
            sig_val = np.sqrt(sigmai * sigmaj) * temp_weeks * temp_years
            Sa_t[i, j] = sig_val
            Sa_t[j, i] = sig_val

    return Sa_t

args = Arguments()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)
    cfg["start_time"] = args.start_time
    cfg["end_time"] = args.end_time

config = TROPOMI_config(cfg)

file = f"{config.output_path}/inversion_data_{config.inv_start_time.strftime('%Y%m%d%H')}_{config.inv_end_time.strftime('%Y%m%d%H')}.nc"
if os.path.exists(file):
    print(f"{file} already exists. Terminating ...")
else:
    pprint(cfg)
    train_df = get_stored_df(config.tropomi_train_file, config.start_time, config.inv_end_time)
    test_df = get_stored_df(config.tropomi_test_file, config.inv_start_time, config.inv_end_time)
    print("Train size:", train_df.shape, "Test size:", test_df.shape)
    
    train_obs_dict = get_obs_dict(train_df)
    test_obs_dict = get_obs_dict(test_df)
    
    bkg_dict, bkg_date_range = load_background_data(config)
    xp_bkg_date_range = list(pd.date_range(start=config.inv_start_time, end=config.inv_end_time, freq="1D"))
    
    dates = pd.date_range(start=config.inv_start_time, end=config.inv_end_time, freq="W")
    
    lons = config.lons
    lats = config.lats
    m = lons.shape[0]*lats.shape[0]
    inventory_type = config.inventory_type
    inventory_path = config.inventory_path
    print(f"inventory type: {inventory_type} | inventory path: {inventory_path}")
    emission = getEmissions(lons, lats, inventory_type, m, inventory_path=inventory_path, ems_scaling_factor=config.ems_scaling_factor)
    print(dates, dates.shape)
    xp = emission.compute_x_prior_vector(dates)
    print("Prior flux shape:", xp.shape)
    
    # Background prior
    xp_bkg, xp_bkg_error = get_bkg_prior_error() 
    
    # Jacobian flux
    H_old_train = fill_H(config, dates, train_obs_dict)
    H_old_test = fill_H(config, dates, test_obs_dict)
    # print(H_old_train)
    print("Jacobian flux train shape:", H_old_train.shape)
    print("Jacobian flux test shape:", H_old_test.shape)
    # import pdb; pdb.set_trace()
    # Jacobian background
    H_b_train = compute_H_background(train_obs_dict, config)
    H_b_test = compute_H_background(test_obs_dict, config)
    print(H_b_train)
    print(H_b_test)
    print("Jacobian background train shape:", H_b_train.shape)
    print("Jacobian background test shape:", H_b_test.shape)
    
    # Observation error correlation matrix
    print("Computing obs error correlation matrix (R)")
    R = fill_R(train_obs_dict)
    # print(R)
    
    # Computing D and E
    print("Computing D and E")
    tau_week = 52
    tau_year = 5
    Sa_t_mod = get_Sa_t(xp, tau_week, tau_year, correlation_type="mod")
    Sa_xy = load_npz("data/Sa_xy.npz")
    
    ems_uncert = config.ems_uncert
    D, E = compute_D_E(Sa_xy, Sa_t_mod, emission, ems_uncert)
    
    # Computing background prior error covariance matrix
    print("Computing background prior error covariance matrix (B_b)")
    def compute_bkg_prior_error_covariance(xp_bkg, xp_bkg_date_range, xp_bkg_error, tau_day_bkg=1, lower_bound=1e-5):
        B_b = np.zeros((xp_bkg.shape[0], xp_bkg.shape[0]))
        for i in range(B_b.shape[0]):
            t_index_i = i // 8
            octant_index_i = i - t_index_i*8
            # print(t_index_i, octant_index_i, i)
            sigmai = xp_bkg_error[octant_index_i, t_index_i]
            B_b[i, i] = sigmai**2
            tstampi = xp_bkg_date_range[t_index_i]
            for j in range(i+1, B_b.shape[0]):
                t_index_j = j // 8
                octant_index_j = j - t_index_j*8
                tstampj = xp_bkg_date_range[t_index_j]
                sigmaj = xp_bkg_error[octant_index_j, t_index_j]
                if octant_index_i == octant_index_j:
                    time_delay = (tstampj - tstampi).days
                    temp_time = np.exp(-time_delay/tau_day_bkg)
                    sig_val = sigmai*sigmaj*temp_time
                    if sig_val >= lower_bound:
                        B_b[i, j] = sig_val
                        B_b[j, i] = sig_val
        return B_b
    B_b = compute_bkg_prior_error_covariance(xp_bkg, xp_bkg_date_range, xp_bkg_error, tau_day_bkg=1, lower_bound=1e-5)
    print(B_b)
    
    
    # Inversion data prep
    Y = np.array([val["methane_mixing_ratio_bias_corrected"] for val in train_obs_dict.values()]).reshape(-1, 1)
    xp_comb = np.concat([xp, xp_bkg], axis=0)
    Y = csc_matrix(Y)
    xp_comb = csc_matrix(xp_comb)
    
    H_b_train_sparse = csc_matrix(H_b_train)
    H_train_comb = sparse.hstack((H_old_train, H_b_train_sparse), format='csc')
    
    # Obs error correlation to covariance
    y_sim = H_train_comb @ xp_comb
    mismatch = Y - y_sim
    sigma_mismatch = np.std(mismatch.toarray())
    
    Ro = R*(sigma_mismatch**2)
    
    # HB and HBHT matrices
    print("Computing HB and HBHT matrices")
    HB_old = HQ_sparse_parallel(H_old_train, D, E, n_jobs=4)
    HBHT_old = HQHT_sparse(HB_old, H_old_train, D, E)
    
    HB_b = H_b_train @ B_b
    HB_b = csc_matrix(HB_b)
    
    HBHT_b = HB_b @ H_b_train.T
    HBHT_b = csc_matrix(HBHT_b)
    
    HB_comb = sparse.hstack((HB_old, HB_b), format='csc')
    HBHT_comb = HBHT_old + HBHT_b
    
    # Conducting analytical inversion
    print("Inversion")
    mismatch = csc_matrix(mismatch)
    xp_comb = csc_matrix(xp_comb)
    
    inv_term = inv(HBHT_comb + Ro)
    gain1 = inv_term @ mismatch
    xdiff = HB_comb.T @ (gain1)
    xpost = xp_comb + xdiff
    
    xpost_fluxes = xpost[:xp.shape[0]]
    xpost_bkg = xpost[xp.shape[0]:]
    
    print("Gathering fluxes")
    xfluxes = np.zeros((dates.shape[0], config.lats.shape[0], config.lons.shape[0]))
    xpost_fluxes = xpost_fluxes.toarray()
    dates_list = []
    for idx, date in enumerate(dates):
        date = int(date.strftime("%Y%m%d%H"))
        print(idx, date)
        dates_list.append(date)
        xfluxes[idx, :, :] = xpost_fluxes[idx*m:(idx+1)*m, 0].reshape(config.lats.shape[0], config.lons.shape[0], order="F")
    
    xpost = xpost.toarray()
    xp_comb = xp_comb.toarray()
    xpost_bkg = xpost_bkg.toarray()
    
    # Evaluation
    Y_test = np.array([val["methane_mixing_ratio_bias_corrected"] for val in test_obs_dict.values()]).reshape(-1, 1)
    H_b_test_sparse = csc_matrix(H_b_test)
    H_test_comb = sparse.hstack((H_old_test, H_b_test_sparse), format='csc')
    
    ye_pred_train = H_old_train @ xpost_fluxes
    ye_prior_train = H_old_train @ xp
    ye_pred_test = H_old_test @ xpost_fluxes
    ye_prior_test = H_old_test @ xp
    
    ysim_pred_train = H_train_comb @ xpost
    ysim_prior_train = H_train_comb @ xp_comb
    ysim_pred_test = H_test_comb @ xpost
    ysim_prior_test = H_test_comb @ xp_comb
    
    ybkg_prior_train = H_b_train @ xp_bkg
    ybkg_pred_train = H_b_train @ xpost_bkg
    ybkg_prior_test = H_b_test @ xp_bkg
    ybkg_pred_test = H_b_test @ xpost_bkg
    
    bkg_prior = xp_bkg.reshape((8, len(xp_bkg_date_range)), order="F")
    xpost_bkg_store = xpost_bkg.reshape((8, len(xp_bkg_date_range)), order="F")
    bkg_post = xpost_bkg_store
    
    
    # Saving results
    print("Storing data")

    out_nc = nc.Dataset(file, "w", format="NETCDF4")
    out_nc.createDimension("train_nobs", Y.shape[0])
    out_nc.createDimension("test_nobs", Y_test.shape[0])
    out_nc.createDimension("lon", config.lons.shape[0])
    out_nc.createDimension("lat", config.lats.shape[0])
    out_nc.createDimension("time", dates.shape[0])
    out_nc.createDimension("bkg_octants", 8)
    out_nc.createDimension("bkg_dates", len(xp_bkg_date_range))
    
    out_nc.createVariable("lat", "f8", ("lat"))[:] = config.lats
    out_nc.createVariable("lon", "f8", ("lon"))[:] = config.lons
    out_nc.createVariable("date", "f8", ("time"))[:] = dates_list
    out_nc.createVariable("bkg_dates", "f8", ("bkg_dates"))[:] = [int(val.strftime("%Y%m%d%H")) for val in xp_bkg_date_range]
    
    out_nc.createVariable("y_actual_train", "f8", ("train_nobs"))[:] = Y.toarray()
    out_nc.createVariable("y_actual_test", "f8", ("test_nobs"))[:] = Y_test
    
    out_nc.createVariable("bkg_prior", "f8", ("bkg_octants", "bkg_dates"))[:, :] = bkg_prior
    out_nc.createVariable("bkg_post", "f8", ("bkg_octants", "bkg_dates"))[:, :] = bkg_post
    
    out_nc.createVariable("ye_pred_train", "f8", ("train_nobs"))[:] = ye_pred_train
    out_nc.createVariable("ye_prior_train", "f8", ("train_nobs"))[:] = ye_prior_train
    out_nc.createVariable("ye_pred_test", "f8", ("test_nobs"))[:] = ye_pred_test
    out_nc.createVariable("ye_prior_test", "f8", ("test_nobs"))[:] = ye_prior_test
    
    out_nc.createVariable("ysim_pred_train", "f8", ("train_nobs"))[:] = ysim_pred_train
    out_nc.createVariable("ysim_prior_train", "f8", ("train_nobs"))[:] = ysim_prior_train
    out_nc.createVariable("ysim_pred_test", "f8", ("test_nobs"))[:] = ysim_pred_test
    out_nc.createVariable("ysim_prior_test", "f8", ("test_nobs"))[:] = ysim_prior_test
    
    
    out_nc.createVariable("ybkg_pred_train", "f8", ("train_nobs"))[:] = ybkg_pred_train
    out_nc.createVariable("ybkg_prior_train", "f8", ("train_nobs"))[:] = ybkg_prior_train
    out_nc.createVariable("ybkg_pred_test", "f8", ("test_nobs"))[:] = ybkg_pred_test
    out_nc.createVariable("ybkg_prior_test", "f8", ("test_nobs"))[:] = ybkg_prior_test
    
    
    out_nc.createVariable("post_fluxes", "f8", ("time", "lat", "lon"))[:, :, :] = xfluxes 
    out_nc.createVariable("prior_fluxes", "f8", ("lat", "lon"))[:, :] = emission.emissions
    out_nc.close()
