import netCDF4 as nc
import glob, os
import argparse
import datetime
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Polygon, Point

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.img_tiles import GoogleTiles
from tqdm import tqdm

from ColumnFootNet import ColumnFootNet
from getBackground import Background
from getColumnMeteorology import ColumnMeteorology

class getTROPOMI():
    def __init__(self, config, bkg_compute=False):
        self.config = config
        self.lons = config.lons
        self.lats = config.lats
        self.footprint_path = config.footprint_path
        self.start_time = config.start_time
        self.end_time = config.end_time
        self.file = config.tropomi_filepath
        self.upwind_lons = [self.lons[0] - config.upwind_degree_margin, self.lons[-1] + config.upwind_degree_margin]
        self.upwind_lats = [self.lats[0] - config.upwind_degree_margin, self.lats[-1] + config.upwind_degree_margin]

        self.upwind_date_margin = config.upwind_date_margin
        self.background_date_range = pd.date_range(start=self.start_time-datetime.timedelta(days=self.upwind_date_margin), end=self.end_time+datetime.timedelta(self.upwind_date_margin), freq="1d")
        print(self.background_date_range)
        self.upwind_boundary = Polygon([
            (self.lons[0] - config.upwind_degree_margin, self.lats[0] - config.upwind_degree_margin),
            (self.lons[0] - config.upwind_degree_margin, self.lats[-1] + config.upwind_degree_margin),
            (self.lons[-1] + config.upwind_degree_margin, self.lats[-1] + config.upwind_degree_margin),
            (self.lons[-1] + config.upwind_degree_margin, self.lats[0] - config.upwind_degree_margin)
        ])
        
        self.flux_boundary = Polygon([(self.lons[0], self.lats[0]), (self.lons[0], self.lats[-1]), (self.lons[-1], self.lats[-1]), (self.lons[-1], self.lats[0])])
        
        self.obs_boundary = Polygon([
            (self.lons[config.obs_indices_margin], self.lats[config.obs_indices_margin]),
            (self.lons[config.obs_indices_margin], self.lats[-config.obs_indices_margin]),
            (self.lons[-config.obs_indices_margin], self.lats[-config.obs_indices_margin]),
            (self.lons[-config.obs_indices_margin], self.lats[config.obs_indices_margin])
            
        ])

        print(f"Filtering tropomi obs including upwind domain ({self.upwind_boundary.exterior.xy})...")
        self.upwind_df = self.find_obs(self.start_time, self.end_time, self.upwind_boundary)
        print(f"Total obs (including upwind): {self.upwind_df.shape[0]}")
        print(f"Filtering tropomi obs in observation domain ({self.obs_boundary.exterior.xy}) ...")
        self.obs_df = self.find_obs(self.start_time, self.end_time, self.obs_boundary, df=self.upwind_df)
        print(f"Total obs in observation domain: {self.obs_df.shape[0]}")

        print(f"Computing TROPOMI subpixels for {self.obs_df.shape[0]} obs ...")
        self.subpixels_df = self.compute_subpixels(self.obs_df, self.lats, self.lons)

        self.octant_dict = self.get_background_octant()
        if bkg_compute:
            self.background = Background(self.obs_df, self.upwind_df, self.octant_dict, self.background_date_range, config)
            self.compute_background()

    def find_obs(self, start_time, end_time, polygon_boundary, df=None):
        if df is None:
            print(f"Loading {self.file} ....")
            df = pd.read_csv(self.file)
            df['lons'] = df['lon_str'].apply(lambda x: [val for val in x.split("|")])
            df['lats'] = df['lat_str'].apply(lambda x: [val for val in x.split("|")])
            df = df.drop(['lon_str', 'lat_str'], axis=1)
            df['geometry'] = df[['lons', 'lats']].apply(lambda x: Polygon(zip(x.iloc[0]+[x.iloc[0][0]], x.iloc[1]+[x.iloc[1][0]])), axis=1)
            df['delta_time'] = df['delta_time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
            df['actual_time'] = df['actual_time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
            df['time'] = df['actual_time'].apply(lambda x:x.round('60min').to_pydatetime())
            df = gpd.GeoDataFrame(df, geometry="geometry")
        
        # dk = df[(self.start_time <= df['actual_time']) & (df['actual_time'] <= self.end_time) & (self.upwind_lons[0] <= df['lon']) & (df['lon'] <= self.upwind_lons[-1]) & (self.upwind_lats[0] <= df['lat']) & (df['lat'] <= self.upwind_lats[-1])].reset_index(drop=True)
        dk = df[(start_time <= df['actual_time']) & (df['actual_time'] <= end_time) & (df.geometry.within(polygon_boundary))].reset_index(drop=True)
        return dk

    def compute_subpixels(self, obs_df, lats, lons):
        subpixel_grid = pd.DataFrame(np.vstack((np.repeat(lats, lons.shape[0]), np.tile(lons, lats.shape[0]))).T, columns=['grid_lat', 'grid_lon'])
        subpixel_grid = gpd.GeoDataFrame(geometry=[Point(subpixel_grid['grid_lon'][idx], subpixel_grid['grid_lat'][idx]) for idx in range(subpixel_grid.shape[0])])
        
        obs_df['gp_geometry'] = obs_df['geometry']
        subpixels_df = gpd.sjoin(obs_df, subpixel_grid, how='right')
        subpixels_df = subpixels_df.dropna().reset_index(drop=True)
        subpixels_df['centroid_lon'] = [val.centroid.x for val in subpixels_df['geometry']]
        subpixels_df['centroid_lat'] = [val.centroid.y for val in subpixels_df['geometry']]
        subpixels_df = subpixels_df.sort_values(by="index_left")
        return subpixels_df

    def createPolygon(self, lat_min, lat_max, lon_min, lon_max):
        """
        Computes the vertices of a rectangular polygon (bounding box)
        given minimum and maximum latitude and longitude values.
    
        Args:
            lat_min (float): The minimum latitude.
            lat_max (float): The maximum latitude.
            lon_min (float): The minimum longitude.
            lon_max (float): The maximum longitude.
    
        Returns:
            list: A list of tuples, where each tuple represents a vertex
                  (longitude, latitude) of the polygon in counter-clockwise order.
        """
        # Vertices in counter-clockwise order: bottom-left, bottom-right, top-right, top-left
        polygon_vertices = [
            (lon_min, lat_min),  # Bottom-left
            (lon_max, lat_min),  # Bottom-right
            (lon_max, lat_max),  # Top-right
            (lon_min, lat_max)   # Top-left
        ]
        return Polygon(polygon_vertices)
    
    def get_background_octant(self):
        clat_index = self.lats.shape[0]//2
        clon_index = self.lons.shape[0]//2
        octant_dict = {}
        octant_dict[1] = self.createPolygon(self.lats[-1], self.upwind_lats[-1], self.lons[clon_index], self.lons[-1])
        octant_dict[2] = self.createPolygon(self.lats[clat_index], self.lats[-1], self.lons[-1], self.upwind_lons[-1])
        octant_dict[3] = self.createPolygon(self.lats[0], self.lats[clat_index], self.lons[-1], self.upwind_lons[-1])
        octant_dict[4] = self.createPolygon(self.upwind_lats[0], self.lats[0], self.lons[clon_index], self.lons[-1])
        octant_dict[5] = self.createPolygon(self.upwind_lats[0], self.lats[0], self.lons[0], self.lons[clon_index])
        octant_dict[6] = self.createPolygon(self.lats[0], self.lats[clat_index], self.upwind_lons[0], self.lons[0])
        octant_dict[7] = self.createPolygon(self.lats[clat_index], self.lats[-1], self.upwind_lons[0], self.lons[0])
        octant_dict[8] = self.createPolygon(self.lats[-1], self.upwind_lats[-1], self.lons[0], self.lons[clon_index])
        return octant_dict

    def compute_background(self):
        bkgs = []
        bkgs_error = []
        self.trajectories = []
        for idx in range(self.obs_df.shape[0]):
            timestamp = self.obs_df['time'][idx]
            centroid_gp = self.obs_df["gp_geometry"][idx].centroid
            rlat, rlon = centroid_gp.y, centroid_gp.x
            bkg, bkg_error, trajectory = self.background.get_background_value(timestamp, rlon, rlat)
            bkgs.append(bkg)
            bkgs_error.append(bkg_error)
            self.trajectories.append(trajectory)
        refs = [val["ref_list"] for val in self.trajectories]
        norm_dist = [val["norm_dist"] for val in self.trajectories]
        self.obs_df["bkg"] = bkgs
        self.obs_df["bkg_error"] = bkgs_error
        self.obs_df["bkg_ref"] = refs
        self.obs_df["norm_dist"] = norm_dist

    def load_footprints(self):
        dict_list = self.obs_df.to_dict(orient='records')
        self.obs_dict = {}
        for idx in tqdm(range(len(dict_list))):
            data = dict_list[idx]
            tstamp = data['time']
            rlon = data['lon']
            rlat = data['lat']
            dir_path = f"{self.footprint_path}/{tstamp.year}/{tstamp.month}"
            filename = f"{dir_path}/footnet_footprint_TROPOMI_GP_{datetime.datetime.strftime(tstamp, '%Y%m%d%H')}_{rlat}_{rlon}.nc"
            ds = nc.Dataset(filename)
            foot = np.array(ds["foot"])
            ds.close()
            data['foot'] = foot
            self.obs_dict[idx] = data

    def compute_footprints(self):
        # Gathering meteorology
        timelist = list(set(self.obs_df["time"]))
        self.input_met_foot = ColumnMeteorology(timelist, self.lons, self.lats, self.config.trimsize, self.config.hr3lat_full, self.config.hr3lon_full, self.config.HRRR_DIR, backhours=[0, 6, 12, 18, 24])
        model = ColumnFootNet(model_path=self.config.model_path)
        print(f"Computing Footprints for {self.subpixels_df.shape[0]} subpixels...")
        self.obs_dict = {}
        grouped = self.subpixels_df.groupby("index_left")
        for idx, data in tqdm(grouped):
            data = data.reset_index(drop=True)
            # timelist = list(set(data['time']))
            # Subpixels as receptors
            assert len(set(data['time'])) == 1
            assert len(set(data['lat'])) == 1
            assert len(set(data['lon'])) == 1
            
            row = self.obs_df[self.obs_df.index==int(idx)].reset_index(drop=True)
            assert row['time'][0] == data['time'][0]
            assert row['lat'][0] == data['lat'][0]
            assert row['lon'][0] == data['lon'][0]
            
            self.obs_dict[int(idx)] = row.to_dict(orient='records')[0]
            tstamp = self.obs_dict[int(idx)]['time']
            dir_path = f"{self.footprint_path}/{tstamp.year}/{tstamp.month}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            filename = f"{dir_path}/footnet_footprint_TROPOMI_GP_{datetime.datetime.strftime(self.obs_dict[int(idx)]['time'], '%Y%m%d%H')}_{self.obs_dict[int(idx)]['lat']}_{self.obs_dict[int(idx)]['lon']}.nc"
            if not os.path.exists(filename):
                receptors = [[data['time'][jdx].to_pydatetime(), float(data['centroid_lon'][jdx]), float(data['centroid_lat'][jdx])] for jdx in range(data.shape[0])]
                # Gather meteorology
                
                # Compute footprints
                foots, reference_indices, reference_timestamps, reference_rlons, reference_rlats, reference_foot_hours = model.run_inference(receptors, self.input_met_foot, maximum_domain_trajectory=self.config.maximum_domain_trajectory)
                
                # print("Row:", row)
                self.obs_dict[int(idx)]['foot'] = np.mean(foots, axis=0)
                self.obs_dict[int(idx)]["avg_transport_hours"] = np.mean(reference_foot_hours)
                self.write_footprint_file(filename, self.obs_dict[int(idx)])
        # print(self.obs_dict)

    def write_footprint_file(self, filename, obs_dict):
        
        out_nc = nc.Dataset(filename, "w", format="NETCDF4")
        out_nc.createDimension("lat", self.lats.shape[0])
        out_nc.createDimension("lon", self.lons.shape[0])
        out_nc.createDimension("info", 1)
        
        lat = out_nc.createVariable("lat", np.float32, ("lat",))
        lon = out_nc.createVariable("lon", np.float32, ("lon",))
        val = out_nc.createVariable("foot", np.float32, ("lat", "lon"))
        rlat = out_nc.createVariable("receptor_lat", np.float32, ("info"))
        rlon = out_nc.createVariable("receptor_lon", np.float32, ("info"))
        lat[:] = self.lats
        lon[:] = self.lons
        val[:, :] = obs_dict["foot"]
        rlat[:] = obs_dict["lat"]
        rlon[:] = obs_dict["lat"]
        out_nc.close()

    def plot_domains(self):
        
        tiler = GoogleTiles(style='satellite')

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)

        x, y = self.upwind_boundary.exterior.xy
        ax.set_extent([np.min(x), np.max(x), np.min(y), np.max(y)], crs=ccrs.PlateCarree())
        ax.add_image(tiler, 10, alpha=0.5)
        
        gl = ax.gridlines(ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='white', alpha=0.5, linestyle='--')
        
        # Control which labels are drawn
        gl.top_labels = False
        gl.right_labels = False
        
        # Use formatters to correctly label the axes as longitudes and latitudes
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        if self.upwind_df.shape[0] > 0:
            timestamp = self.upwind_df['time'][0]
            for gp in self.upwind_df[self.upwind_df['time']==timestamp]['geometry']:
                x, y = gp.exterior.xy
                ax.plot(x, y, color='k', alpha=0.5, linewidth=1, transform=ccrs.PlateCarree())
            ax.set_title(f"{timestamp} UTC", fontsize=15)
        
        x, y = self.flux_boundary.exterior.xy
        ax.plot(x, y, color="k", linewidth=2, linestyle="-", transform=ccrs.PlateCarree())
        
        x, y = self.obs_boundary.exterior.xy
        ax.plot(x, y, color="k", linewidth=2, linestyle="-", transform=ccrs.PlateCarree())

        for octant in self.octant_dict:
            x, y = self.octant_dict[octant].exterior.xy
            ax.plot(x, y, color="k", linewidth=2, linestyle="-", transform=ccrs.PlateCarree())
            
        plt.tight_layout()
        fig.savefig("domain.png", dpi=300)
        plt.show()



class TROPOMI_config():
    def __init__(self, cfs):
        self.tropomi_filepath = cfs["tropomi_filepath"]
        self.start_time = datetime.datetime.strptime(cfs["start_time"], "%Y%m%d%H")
        self.end_time = datetime.datetime.strptime(cfs["end_time"], "%Y%m%d%H")
    
        self.xres = cfs["xres"]
        self.yres = cfs["yres"]
        self.clon = cfs["clon"]
        self.clat = cfs["clat"]
        # ax.scatter([clon], [clat])
        # self.lons = np.arange(self.clon-200*self.xres, self.clon+200*self.xres+0.001, self.xres)[:-1]
        # self.lats = np.arange(self.clat-200*self.xres, self.clat+200*self.xres+0.001, self.xres)[:-1]
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
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="PATH to YAML config file")
    parser.add_argument("--start_time", required=True, help="Inversion start date")
    parser.add_argument("--end_time", required=True, help="Inversion end date")
    args = parser.parse_args()
    print(args)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["start_time"] = args.start_time
        cfg["end_time"] = args.end_time
    print(cfg)
    config = TROPOMI_config(cfg)
    print(config)
    tropomi = getTROPOMI(config)
    print(tropomi.subpixels_df)
    tropomi.plot_domains()
    tropomi.compute_footprints()
    