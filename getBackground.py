import netCDF4 as nc
from tqdm import tqdm
from scipy.spatial import Delaunay
import datetime
import numpy as np
import pandas as pd
from shapely import Polygon, Point
from math import sin, cos, asin, atan2, radians, degrees
from joblib import Parallel, delayed

def interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out, d=2):
    xy=np.zeros([grid_x_in.shape[0]*grid_x_in.shape[1],2])
    uv=np.zeros([grid_x_out.shape[0]*grid_x_out.shape[1],2])
    xy[:,0] = grid_x_in.flatten('F')
    xy[:,1] = grid_y_in.flatten('F')
    uv[:,0] = grid_x_out.flatten('F')
    uv[:,1] = grid_y_out.flatten('F')
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def regmet(data, vtx, wts, out_dims):
    '''
    Read and regrid met fields to given longitudes/latitudes
    '''
    data_out = interpolate(data.flatten('F'),vtx,wts).reshape(out_dims, order='F')
    return data_out


def get_hrrr_file(yy, mm, dd, hh, HRRR_DIR):
    # 0, 6, 12, 18
    hhh = [0, 6, 12, 18]
    hidx = int(hh//6)
    return HRRR_DIR + '%04d/hysplit.%04d%02d%02d.%02dz.nc'%(yy, yy, mm, dd, hhh[hidx])
    # return HRRR_DIR + 'hysplit.%04d%02d%02d.%02dz.hrrra'%(yy, mm, dd, hhh[hidx])


def get_winds(footlons, footlats, timestamp, HRRR_DIR, trimsize, hr3lat_full, hr3lon_full, hist=0):
    predlist = ['U10M', 'V10M'] # 2
    reftime = datetime.datetime(1950, 1, 1, 0, 0, 0, 0)
    
    clon = footlons[int(footlons.shape[0]/2)]
    clat = footlats[int(footlats.shape[0]/2)]
    dtnow = datetime.datetime.strptime(timestamp[:10], "%Y%m%d%H")
    histdt = dtnow + datetime.timedelta(hours=hist)
    _yy, _mm, _dd, _hh = histdt.year, histdt. month, histdt.day, histdt.hour
    h3rfile = get_hrrr_file(_yy, _mm, _dd, _hh, HRRR_DIR)
    
    fh = nc.Dataset(h3rfile)
    # fh = xr.open_dataset(h3rfile, engine='pseudonetcdf')
    # fh = fh_dict[histdt]
    h3r_data = fh.variables
    
    
    # times = fh.coords['time'].values
    times = [pd.to_datetime(int(val), unit='ns') for val in np.array(fh['time'])]
    # times = pd.to_datetime(times)
    # tdelt = (times - histdt).seconds
    tidx = np.argmin(np.abs(np.array(times) - histdt))
    # print(h3rfile, tidx, histdt)
    
    distances = (hr3lon_full - clon)**2 + (hr3lat_full - clat)**2
    cind = np.argwhere(distances == np.min(distances))[0]
    cxind = cind[0] # lat
    cyind = cind[1] # lon
    hr3lon = hr3lon_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    hr3lat = hr3lat_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    
    # Regridding weights
    grid_x_in, grid_y_in = hr3lon, hr3lat
    grid_x_out, grid_y_out = np.meshgrid(footlons, footlats)
    vtx, wts = interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out)
    
    _u10m = np.array(h3r_data['U10M'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
    _u10mr = regmet(_u10m, vtx, wts, grid_x_out.shape)
    
    _v10m = np.array(h3r_data['V10M'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize])
    _v10mr = regmet(_v10m, vtx, wts, grid_x_out.shape)
    
    output = np.zeros((footlons.shape[0], footlats.shape[0], len(predlist)))
    output[:, :, 0] = _u10mr
    output[:, :, 1] = _v10mr
    return output

class Background():
    def __init__(self, dk, domain_gdf, octant_dict, background_date_range, config):
        self.timestamp_list = list(set(dk['time']))
        self.met_dict = {}
        self.HRRR_DIR = config.HRRR_DIR
        self.trimsize = config.trimsize
        self.lats = config.lats
        self.lons = config.lons
        self.hr3lat_full = config.hr3lat_full
        self.hr3lon_full = config.hr3lon_full
        self.background_date_range = background_date_range
        self.met_temp_resolution = config.met_temp_resolution_background
        self.fetch_met_data()
            
        self.background_dict = {}
        for octant in octant_dict:
            print("Octant:", octant)
            self.background_dict[octant] = {}
            self.background_dict[octant]['bkg'] = [None for i in range(self.background_date_range.shape[0])]
            self.background_dict[octant]['bkg_error'] = [None for i in range(self.background_date_range.shape[0])]
            self.background_dict[octant]['count'] = []
            self.background_dict[octant]["boundary"] = octant_dict[octant]
            for jdx, date in enumerate(self.background_date_range):
                concs = list(domain_gdf[(domain_gdf['reference_time']==datetime.datetime.strftime(date, "%Y-%m-%d")) & (domain_gdf.geometry.within(octant_dict[octant]))]['methane_mixing_ratio_bias_corrected'])
                
                self.background_dict[octant]['count'].append(len(concs))
                if concs:
                    self.background_dict[octant]['bkg'][jdx] = np.average(concs)
                    self.background_dict[octant]['bkg_error'][jdx] = np.std(concs)
                    # self.background_dict[octant]['bkg_error'][jdx] = np.std(concs, ddof=1)/len(concs) # Computing standard deviation of the mean

            self.background_dict[octant]['bkg'], self.background_dict[octant]['bkg_error'] = self.fill_missing_values(self.background_dict[octant]['bkg'], self.background_dict[octant]['bkg_error'], background_date_range)

    def fill_missing_values(self, bkg, bkg_error, background_date_range):
        bkg = list(pd.Series(bkg, index=background_date_range).interpolate(method='linear').ffill().bfill())
        bkg_error = pd.Series(bkg_error, index=background_date_range)
        bkg_error[bkg_error==0] = None
        bkg_error = list(bkg_error.interpolate(method='linear').ffill().bfill())
        return bkg, bkg_error

    def fetch_met_data(self):
        OUTPUT = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(delayed(self.get_met_data)(timestamp) for timestamp in self.timestamp_list)
        for ref in OUTPUT:
            for key in ref:
                if key not in self.met_dict:
                    self.met_dict[key] = ref[key]
        
    
    def get_met_data(self, timestamp):
        met_dict = {}
        met_timestamps = [timestamp + datetime.timedelta(hours=-i*self.met_temp_resolution) for i in range(0, 72)]
        for met_timestamp in met_timestamps:
            if met_timestamp not in met_dict:
                met_dict[met_timestamp] = get_winds(self.lons, self.lats, datetime.datetime.strftime(met_timestamp, "%Y%m%d%H"), self.HRRR_DIR, self.trimsize, self.hr3lat_full, self.hr3lon_full)
        return met_dict

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
        return degrees(lat2), degrees(lon2)
    
    def get_distance(self, rlon, rlat, lon, lat):
        """
        Calculate the distance from (rlon, rlat) to all points in (lon, lat) grid
        using the haversine formula.
        rlon, rlat: reference longitude and latitude
        lon, lat: 2D arrays of longitudes and latitudes
        """
        
        phi1, phi2 = np.radians(lat), np.radians(rlat)
        dphi = np.radians(lat-rlat)
        dlambda = np.radians(lon-rlon)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)* (np.sin(dlambda/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dist = 6371 * c
        return dist

    def get_weighted_background(self, exit_lat, exit_lon, exit_time):
        exit_time = exit_time.round('60min').to_pydatetime()
        timestamp = exit_time - datetime.timedelta(hours=exit_time.hour)
        # print(exit_time, timestamp)
        index = list(self.background_date_range).index(timestamp)
        bkg = 0.0
        bkg_error = 0.0
        norm_dist = 0.0
        ref_list = []
        for octant in self.background_dict:
            boundary = self.background_dict[octant]["boundary"]
            bkg_octant = self.background_dict[octant]["bkg"][index]
            bkg_error_octant = self.background_dict[octant]["bkg_error"][index]
            dist = self.get_distance(exit_lon, exit_lat, boundary.centroid.x, boundary.centroid.y)
            bkg +=  bkg_octant / dist
            bkg_error += bkg_error_octant / dist
            norm_dist += 1/dist
            ref_list.append([octant, dist, exit_time, bkg_octant, bkg_error_octant])
        bkg = bkg/norm_dist
        bkg_error = bkg_error/norm_dist

        return bkg, bkg_error, norm_dist, ref_list

    def get_background_value(self, tstamp, rlon, rlat):
        met_timestamps = [tstamp + datetime.timedelta(hours=-i*self.met_temp_resolution) for i in reversed(range(1, 72))]
        u_list, v_list = [], []
        for met_timestamp in met_timestamps:
            met_data = self.met_dict.get(met_timestamp, None)
            if met_data is not None:
                u, v = np.average(met_data[:, :, 0]), np.average(met_data[:, :, 1])
                # Reversed winds
                u_list.append(-u)
                v_list.append(-v)

        trajectory = self.get_trajectory_until_exit(u_list, v_list, self.lats, self.lons, rlon, rlat, dt_hours=self.met_temp_resolution)
        exit_time = trajectory["exit_time"] # hours after release that particles left the domain
        exit_time = tstamp - datetime.timedelta(hours=exit_time)
        exit_indices = (trajectory["exit_lat_idx"], trajectory["exit_lon_idx"])
        bkg, bkg_error, norm_dist, ref_list = self.get_weighted_background(trajectory["exit_lat"], trajectory["exit_lon"], exit_time)
        trajectory["norm_dist"] = norm_dist
        trajectory["ref_list"] = ref_list
        return bkg, bkg_error, trajectory

    

    def get_trajectory_until_exit(self, uxy_list, vxy_list, lats, lons, x_rlon, x_rlat, dt_hours=1):
        """
        Compute particle trajectory given time-varying winds every dt_hours.
        If particle doesn't leave domain, extend last wind until it does.
        """
        grid_lat_min, grid_lat_max = np.min(lats), np.max(lats)
        grid_lon_min, grid_lon_max = np.min(lons), np.max(lons)
    
        x, y = x_rlon, x_rlat
        traj_lons, traj_lats = [x], [y]
    
        exit_lat, exit_lon = None, None
        exit_step = None
    
        dt = dt_hours * 3600  # seconds
    
        winds = list(zip(uxy_list, vxy_list))
    
        step_idx = 0
        while True:
            # pick wind for this step
            if step_idx < len(winds):
                uxy, vxy = winds[step_idx]
            else:
                uxy, vxy = winds[-1]  # extend last wind
    
            # distances in km
            dist_x_km = abs(uxy * dt) / 1000.0
            dist_y_km = abs(vxy * dt) / 1000.0
    
            # determine bearings
            bearing_x = None
            if uxy > 0: bearing_x = 90
            elif uxy < 0: bearing_x = 270
    
            bearing_y = None
            if vxy > 0: bearing_y = 0
            elif vxy < 0: bearing_y = 180
    
            # move in longitude
            if bearing_x is not None:
                y_tmp, x_tmp = self.get_point_at_distance(y, x, dist_x_km, bearing_x)
            else:
                x_tmp, y_tmp = x, y
    
            # move in latitude
            if bearing_y is not None:
                y_new, x_new = self.get_point_at_distance(y_tmp, x_tmp, dist_y_km, bearing_y)
            else:
                x_new, y_new = x_tmp, y_tmp
    
            # check exit
            if not (grid_lat_min <= y_new <= grid_lat_max and grid_lon_min <= x_new <= grid_lon_max):
                # linear interpolation fraction to boundary
                frac_lat = frac_lon = 1.0
                if y_new > grid_lat_max: frac_lat = (grid_lat_max - y) / (y_new - y)
                elif y_new < grid_lat_min: frac_lat = (grid_lat_min - y) / (y_new - y)
                if x_new > grid_lon_max: frac_lon = (grid_lon_max - x) / (x_new - x)
                elif x_new < grid_lon_min: frac_lon = (grid_lon_min - x) / (x_new - x)
    
                frac = min(frac_lat, frac_lon)
    
                exit_lat = y + frac * (y_new - y)
                exit_lon = x + frac * (x_new - x)
    
                traj_lats.append(exit_lat)
                traj_lons.append(exit_lon)
                exit_step = step_idx + frac
                exit_time = exit_step * dt_hours
                break
    
            # normal update
            x, y = x_new, y_new
            traj_lons.append(x)
            traj_lats.append(y)
    
            step_idx += 1
    
        # nearest grid indices
        exit_lat_idx = int(np.clip(np.abs(lats - exit_lat).argmin(), 0, len(lats) - 1))
        exit_lon_idx = int(np.clip(np.abs(lons - exit_lon).argmin(), 0, len(lons) - 1))
    
        return {
            "trajectory_lats": traj_lats,
            "trajectory_lons": traj_lons,
            "exit_lat": exit_lat,
            "exit_lon": exit_lon,
            "exit_lat_idx": exit_lat_idx,
            "exit_lon_idx": exit_lon_idx,
            "exit_step": exit_step,
            "exit_time": exit_time,
            "winds": (uxy_list, vxy_list),
            "coordinates": (traj_lons, traj_lats)
        }

