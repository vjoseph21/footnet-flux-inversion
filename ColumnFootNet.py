import warnings
warnings.filterwarnings('ignore')

import torch
import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
from math import asin, atan2, cos, degrees, radians, sin
from tqdm import tqdm
from unetpp_model import NestedUNet

class ColumnFootDataset(Dataset):
    """
    Dataset class for ColumnFootNet model
    """
    def __init__(self, data, input_met, maximum_domain_trajectory):
        """
        Initialize dataset with data and input meteorology
        data: list of (timestamp, rlon, rlat)
        input_met: ColumnMeteorology object
        """
        self.data = data
        self.lons = input_met.lons
        self.lats = input_met.lats
        self.input_met = input_met
        
        self.maximum_domain_trajectory = maximum_domain_trajectory

    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self.data)
    
    def get_distance(self, rlon, rlat, lat, lon):
        """
        Calculate the distance from (rlon, rlat) to all points in (lon, lat) grid
        using the haversine formula.
        rlon, rlat: reference longitude and latitude
        lon, lat: 2D arrays of longitudes and latitudes
        """
        rlat_index = np.unravel_index((np.abs(lat- rlat)).argmin(), lat.shape)
        rlon_index = np.unravel_index((np.abs(lon- rlon)).argmin(), lon.shape)
        lon, lat = np.meshgrid(lon, lat)
        lon = lon*np.pi/180
        lat = lat*np.pi/180
        rlon = rlon*np.pi/180
        rlat = rlat*np.pi/180
        a = np.sin((lat-rlat)/2)**2 + np.cos(lat)*np.cos(rlat)*(np.sin((lon-rlon)/2)**2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = 6371e3*c/1000
        return d
    
    def get_point_at_distance(self, lat1, lon1, d, bearing, R=6371):
        """
        lat: initial latitude, in degrees
        lon: initial longitude, in degrees
        d: target distance from initial
        bearing: (true) heading in degrees
        R: optional radius of sphere, defaults to mean radius of earth
    
        Returns new lat/lon coordinate {d}km from initial, in degrees
        """
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        a = radians(bearing)
        lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
        lon2 = lon1 + atan2(
            sin(a) * sin(d/R) * cos(lat1),
            cos(d/R) - sin(lat1) * sin(lat2)
        )
        return (degrees(lat2), degrees(lon2),)

    def GaussianPlume(self, lon, lat, fLon, fLat, uu, vv, aA=104, aB=213, wA=6, wB=2):
        '''
        Function to generate a Gaussian plume using wind fields.
        lon, lat: 2D arrays of longitudes and latitudes
        fLon, fLat: source location (longitude, latitude)
        uu, vv: wind components (u, v) in m/s
        aA, aB, wA, wB: Gaussian plume parameters
        '''
        # Grid info
        nX,nY = lon.shape[1], lon.shape[0]
        c     = np.zeros([nY,nX],dtype=float)
        # Windspeed and direction
        wspd = np.sqrt(uu**2.+vv**2.)   # 2D
        wdir = np.arctan2(vv, uu)
        # Parameters and stability class
        x0     = 1e3
        aA, wA = 104., 6.
        aB, wB = 213., 2.
        a      = (wspd - wA)/(wB - wA)*(aB - aA) + aA
        if a < aA:
            a = aA
        if a > aB:
            a = aB
        # Flatten the matrices
        # lon,lat = np.meshgrid(lon,lat)
        out_dim = c.shape
        xx      = (lon - fLon)/120.*1e3
        yy      = (lat - fLat)/120.*1e3
        r       = np.sqrt(xx**2.+yy**2.)
        phi     = np.arctan2(yy,xx)-wdir
        lx      = r*np.cos(phi)
        ly      = r*np.sin(phi)
        sig     = a*(lx/x0)**0.894
        # import pdb; pdb.set_trace()
        # print("GP:", sig, wspd, ly)
        c = 1./(sig*wspd) * np.exp(-0.5 * (ly/sig)**2. )
        
        c = np.ma.masked_array(c)
        c = c.filled(fill_value=0)
        
        c[np.where(np.isnan(c))] = 0.
        # print(c)
        return c
        
    def get_gaussian_plume_inputs_single_pixel(self, uxy_list, vxy_list, x_rlon, x_rlat):
        """
        Generate Gaussian plume inputs for a single pixel over multiple time steps
        uxy_list: list of u wind components at different times
        vxy_list: list of v wind components at different times
        x_rlon, x_rlat: source location (longitude, latitude)
        Returns combined plume and separate plumes for each time step
        """
        grid_x_out, grid_y_out = np.meshgrid(self.lons, self.lats)
        x = x_rlon
        y = x_rlat
        xlist = []
        ylist = []
        
        gp_dict = {}
        for idx in range(len(uxy_list)):
            uxy = uxy_list[idx]
            vxy = vxy_list[idx]
            xlist.append(x)
            ylist.append(y)
        
            gprs = self.GaussianPlume(grid_x_out,grid_y_out,x, y, -uxy, -vxy)
            gp_dict[idx] = gprs
            
            if -uxy > 0: # longitude increases
                # print("longitude increases")
                bearing = 90
                distance = abs(uxy*6*3600/1000)
                x = self.get_point_at_distance(y, x, distance, bearing=bearing)[1]
            elif -uxy < 0: # longitude decreases
                # print("longitude decreases")
                bearing = 270
                distance = abs(uxy*6*3600/1000)
                x = self.get_point_at_distance(y, x, distance, bearing=bearing)[1]
            
            if -vxy >0: # latitude increases
                # print("latitude increases")
                bearing = 0
                distance = abs(vxy*6*3600/1000)
                y = self.get_point_at_distance(y, x, distance, bearing=bearing)[0]
            elif -vxy < 0: # latitude decreases
                # print("latitude decreases")
                bearing = 180
                distance = abs(vxy*6*3600/1000)
                y = self.get_point_at_distance(y, x, distance, bearing=bearing)[0]
        gp = np.zeros((self.lons.shape[0], self.lats.shape[0]))
        gp_separate = np.zeros((self.lons.shape[0], self.lats.shape[0], 5))
        for key in gp_dict:
            gp += gp_dict[key]
            gp_separate[:, :, key] = gp_dict[key]
        return gp, gp_separate
    
    def zstandard(self, arr):
        """
        z-standardize the input array
        arr: input array
        Returns z-standardized array
        """
        _mu = np.nanmean(arr)
        _std = np.nanstd(arr)
        return (arr - _mu)/_std

    def estimate_foot_hours(self, uxy_list, vxy_list, maximum_domain_trajectory, footnet_hours_mode='average'):
        if footnet_hours_mode == 'average':
            u = np.average(uxy_list)
            v = np.average(vxy_list)
            dist_per_hour = np.sqrt((u*3600/1000)**2 + (v*3600/1000)**2) # km
            foot_hours = maximum_domain_trajectory/dist_per_hour # hours
            foot_hours = round(foot_hours)
        return foot_hours

    def __getitem__(self, index):
        """
        Process and return a single sample from the dataset
        index: index of the sample to retrieve
        Returns a tuple (input_tensor, [index, timestamp_str, rlon, rlat]) -> input to FootNet model
        input_tensor: processed input data tensor
        [index, timestamp_str, rlon, rlat]: metadata about the sample for reference
        """
        # print("Index:", idx)
        timestamp, rlon, rlat = self.data[index]
        timstamp_str = datetime.datetime.strftime(timestamp, "%Y%m%d%H")
        dist = self.get_distance(rlon, rlat, self.lats, self.lons)
        tstamp_list = [timestamp+datetime.timedelta(hours=-hist) for hist in [0, 6, 12, 18, 24]]
        preds = []
        uxy_list = []
        vxy_list = []
        for tstamp in tstamp_list:
            dt_str = datetime.datetime.strftime(tstamp, "%Y%m%d%H")
            temp = self.input_met.processed_met_dict[dt_str]
            preds.append(temp)
            tempu = np.average(self.input_met.input_met_dict[dt_str][:, :, 0])
            tempv = np.average(self.input_met.input_met_dict[dt_str][:, :, 1])
            uxy_list.append(tempu)
            vxy_list.append(tempv)

        comb_plume, gp_separate = self.get_gaussian_plume_inputs_single_pixel(uxy_list, vxy_list, rlon, rlat)
        gp_first = gp_separate[:, :, 0]
        gp_first = self.zstandard(gp_first)[:, :, np.newaxis]
        comb_plume = np.array(comb_plume)[:, :, np.newaxis]
        comb_plume[np.where(comb_plume>=0.08)] = 1
        comb_plume[np.where(comb_plume<0.08)] = 0
        # print(gp_first.shape, preds[0].shape, comb_plume.shape, dist.shape)
        tempx = np.concatenate([gp_first] + preds + [comb_plume, dist[:, :, np.newaxis], np.exp(0.01*dist)[:, :, np.newaxis]], axis=2)
        
        tempxx = np.zeros((tempx.shape[2], tempx.shape[0], tempx.shape[1]))
        for idx in range(tempx.shape[2]):
            tempxx[idx, :, :] = tempx[:, :, idx]

        foot_hours = self.estimate_foot_hours(uxy_list, vxy_list, maximum_domain_trajectory=self.maximum_domain_trajectory)
        return tempxx, [index, timstamp_str, rlon, rlat, foot_hours]


class ColumnFootNet():
    """
    ColumnFootNet model class for loading model and running inference
    1. Load the pre-trained model
    2. Run inference on the input data
    3. Post-process the model output
    4. Return the processed output along with reference metadata
    """
    def __init__(self, model_path):
        """
        Initialize the ColumnFootNet model with given parameters.
        model_path: str, path to the pre-trained model file
        """
        self.input_channels = 49
        self.n_classes = 1
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.epsilon = 1e-4
        self.model = NestedUNet(input_channels=self.input_channels, num_classes=self.n_classes).to(self.device)
        self.load_model(self.model, model_path)

    def load_model(self, model, filename):
        """
        Load the pre-trained model from the specified file.
        model: torch.nn.Module, the model to load the state into
        filename: str, path to the model file
        """
        print(f"...Loading {filename}")
        checkpoint = torch.load(filename, weights_only=False, map_location=self.device)
        model.load_state_dict(checkpoint['MODEL_STATE'])

    def post_processing(self, pred):
        """
        Post-process the model predictions to convert them back to original scale.
        pred: torch.Tensor, model predictions
        Returns post-processed predictions as numpy array
        """
        epsilon = self.epsilon
        pred = pred/1000
        pred = pred + np.log(epsilon)
        pred = np.exp(pred) - epsilon
        pred[np.where(pred<0)] = 0
        pred[np.where(pred<5e-8)] = 0
        return pred
    def post_processing_torch(self, pred):
        epsilon = self.epsilon
        pred = pred/1000
        pred = pred + torch.log(torch.tensor(epsilon, device=pred.device))
        pred = torch.exp(pred) - epsilon
        # Clamp values to zero
        pred = torch.clamp(pred, min=0.0)
        # Optional: remove very small values < 5e-8
        pred = torch.where(pred < 5e-8, torch.tensor(0.0, device=pred.device), pred)
        # pred[torch.where(pred<5e-8)] = 0
        return pred
    
    def run_inference(self, receptors, input_met, maximum_domain_trajectory=None):
        """
        Run inference on the input data using the pre-trained model.
        receptors: list of (timestamp, rlon, rlat) tuples
        input_met: ColumnMeteorology object containing input meteorology data
        Returns:
        foots: numpy array of footprints corresponding to receptors
        reference_indices: numpy array of indices corresponding to receptors
        reference_timestamps: list of timestamps corresponding to receptors
        reference_rlons: numpy array of longitudes corresponding to receptors
        reference_rlats: numpy array of latitudes corresponding to receptors
        """
        batch_DG = DataLoader(ColumnFootDataset(receptors, input_met, maximum_domain_trajectory=maximum_domain_trajectory),  batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
        self.model.eval()
        prediction_list = []
        reference_indices = []
        reference_timestamps = []
        reference_rlons = []
        reference_rlats = []
        reference_foot_hours = []
        with torch.no_grad():
            for idx, data in tqdm(enumerate(batch_DG)):
                # import pdb; pdb.set_trace()
                inputs, reference = data
                inputs = inputs.to(self.device, dtype=torch.float)
                prediction = self.model(inputs)
                prediction = self.post_processing_torch(prediction)
                prediction_list.append(prediction.cpu())
                reference_indices.append(reference[0].cpu())
                reference_timestamps += reference[1]
                reference_rlons.append(reference[2].cpu())
                reference_rlats.append(reference[3].cpu())
                reference_foot_hours.append(reference[4].cpu())
                # if idx == 0:
                #     prediction_list = prediction.cpu()
                #     reference_indices = reference[0].cpu()
                #     reference_timestamps = reference[1]
                #     reference_rlons = reference[2].cpu()
                #     reference_rlats = reference[3].cpu()
                #     reference_foot_hours = reference[4].cpu()
                # else:
                #     prediction_list = torch.cat([prediction_list, prediction.cpu()], axis=0)
                #     reference_indices = torch.cat([reference_indices, reference[0].cpu()], axis=0)
                #     reference_timestamps += reference[1]
                #     reference_rlons = torch.cat([reference_rlons, reference[2].cpu()], axis=0)
                #     reference_rlats = torch.cat([reference_rlats, reference[3].cpu()], axis=0)
                #     reference_foot_hours = torch.cat([reference_foot_hours, reference[4].cpu()], axis=0)
        foots = torch.cat(prediction_list, dim=0).cpu().detach().numpy()[:, 0, :, :]
        reference_indices = torch.cat(reference_indices, dim=0).cpu().detach().numpy()
        reference_rlons = torch.cat(reference_rlons, dim=0).cpu().detach().numpy()
        reference_rlats = torch.cat(reference_rlats, dim=0).cpu().detach().numpy()
        reference_foot_hours = torch.cat(reference_foot_hours, dim=0).cpu().detach().numpy()
        # print("Inference done")
        # foots = self.post_processing_torch(prediction_list)[:, 0, :, :].cpu().detach().numpy()
        # print("postprocessing done")
        # reference_indices = reference_indices.cpu().detach().numpy()
        # reference_rlons = reference_rlons.cpu().detach().numpy()
        # reference_rlats = reference_rlats.cpu().detach().numpy()
        # reference_foot_hours = reference_foot_hours.cpu().detach().numpy()
        return foots, reference_indices, reference_timestamps, reference_rlons, reference_rlats, reference_foot_hours

if __name__=="__main__":
    from getColumnMeteorology import ColumnMeteorology
    xres = 1/120
    yres = 1/120
    lons = np.arange(-99.16, -95.83, xres)
    lats = np.arange(30.83, 34.16, yres)
    hr3lon_full = np.load('/home/hermesData/hermes/nd349/footprint_unet/HRRR_lon_lat.npz')['lon']
    hr3lat_full = np.load('/home/hermesData/hermes/nd349/footprint_unet/HRRR_lon_lat.npz')['lat']
    hr3lon_full = (hr3lon_full+180)%360-180  # convert from 0~360 to -180~180
    HRRR_DIR = '/home/hermesData/hermes2/nd349/data/met_data/xstilt_CONUS_data_lite/'
    trimsize = 150

    receptors = [[datetime.datetime(2020, 2, 12, 17), -97.68, 32.438]]
    timelist = list(set([val[0] for val in receptors]))
    input_met = ColumnMeteorology(timelist, lons, lats, trimsize, hr3lat_full, hr3lon_full, HRRR_DIR, backhours=[0, 6, 12, 18, 24])

    # Loading FootNet model
    model_path = "/home/disk/p/nd349/nikhil.dadheech/pointSources/footnet/CONUS/XSTILT/CONUS_300k/Xmodels/XNestedUNet24h_fullrun/best_modelXNestedUNet24h_fullrun.pth"
    model = ColumnFootNet(model_path=model_path)
    foots, reference_indices, reference_timestamps, reference_rlons, reference_rlats, reference_foot_hours = model.run_inference(receptors, input_met)
    print(np.max(foots), reference_rlons, reference_rlats, reference_foot_hours)