import netCDF4 as nc
import glob, os
import argparse
import datetime
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Polygon, Point
import logging
import time
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.img_tiles import GoogleTiles
from tqdm import tqdm

from ColumnFootNet import ColumnFootNet
from getBackground import Background
from getColumnMeteorology import ColumnMeteorology

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class getTROPOMI():
    def __init__(self, config, bkg_compute=True):
        logger.info("="*60)
        logger.info("Initializing TROPOMI processor")
        logger.info("="*60)

        self.config = config
        self.lons = config.lons
        self.lats = config.lats
        self.footprint_path = config.footprint_path
        self.start_time = config.start_time
        self.end_time = config.end_time
        self.file = config.tropomi_filepath
        self.upwind_lons = [self.lons[0] - config.upwind_degree_margin, self.lons[-1] + config.upwind_degree_margin]
        self.upwind_lats = [self.lats[0] - config.upwind_degree_margin, self.lats[-1] + config.upwind_degree_margin]

        logger.info(f"Domain: {self.lons[0]:.2f}°E to {self.lons[-1]:.2f}°E, "
                   f"{self.lats[0]:.2f}°N to {self.lats[-1]:.2f}°N")
        logger.info(f"Inversion period: {self.start_time} to {self.end_time}")
        logger.info(f"TROPOMI file: {self.file}")

        self.upwind_date_margin = config.upwind_date_margin
        self.background_date_range = pd.date_range(start=self.start_time-datetime.timedelta(days=self.upwind_date_margin), end=self.end_time+datetime.timedelta(self.upwind_date_margin), freq="1d")
        logger.info(f"Background date range: {self.background_date_range[0].date()} to {self.background_date_range[-1].date()}")
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

        logger.info("Filtering TROPOMI observations...")
        # Include upwind_date_margin for background estimation spin-up
        upwind_date_margin = config.upwind_date_margin
        upwind_start = self.start_time - datetime.timedelta(days=upwind_date_margin)
        upwind_end = self.end_time + datetime.timedelta(days=upwind_date_margin)
        self.upwind_df = self.find_obs(upwind_start, upwind_end, self.upwind_boundary)
        logger.info(f"✓ Found {self.upwind_df.shape[0]} observations in upwind domain")
        logger.info(f"  DEBUG: upwind_df date range: {self.upwind_df['time'].min()} to {self.upwind_df['time'].max()}")

        logger.info("Filtering to observation domain...")
        self.obs_df = self.find_obs(self.start_time, self.end_time, self.obs_boundary, df=self.upwind_df)
        logger.info(f"✓ Found {self.obs_df.shape[0]} observations in observation domain")

        logger.info(f"Computing subpixels for {self.obs_df.shape[0]} observations...")
        start_time = time.time()
        self.subpixels_df = self.compute_subpixels(self.obs_df, self.lats, self.lons)
        logger.info(f"✓ Subpixel computation completed in {time.time()-start_time:.1f}s")

        self.octant_dict = self.get_background_octant()
        if bkg_compute:
            self.background = Background(self.obs_df, self.upwind_df, self.octant_dict, self.background_date_range, config)
            self.compute_background()
            self.save_background(config)
            # NOTE: create_and_save_train_test_data is called AFTER compute_footprints()
            # because it needs obs_dict which is only created during footprint computation

    def save_background(self, config):
        """Save background dictionary and date range to pickle file"""
        bkg_data = {
            "background_dict": self.background.background_dict,
            "background_date_range": self.background_date_range
        }
        os.makedirs(os.path.dirname(config.tropomi_bkg_file), exist_ok=True)
        with open(config.tropomi_bkg_file, 'wb') as f:
            pickle.dump(bkg_data, f)
        logger.info(f"✓ Background data saved to {config.tropomi_bkg_file}")

        # Add bkg_ref to observation domain dataframe (convert numpy types for CSV serialization)
        bkg_refs_converted = []
        for val in self.trajectories:
            ref_list = val["ref_list"] if val is not None else []
            # Convert numpy types to Python native types for CSV serialization
            if ref_list:
                ref_list = [[item[0], float(item[1]), item[2], float(item[3]), float(item[4])] for item in ref_list]
            bkg_refs_converted.append(ref_list)
        self.obs_df["bkg_ref"] = bkg_refs_converted
        logger.info(f"✓ Added bkg_ref to {len(self.obs_df)} observation domain observations")

    def create_and_save_train_test_data_with_footprints(self, config):
        """Create train/test using ONLY observations with successful footprints"""
        logger.info("="*60)
        logger.info("Creating train/test datasets (footprints only)")
        logger.info("="*60)

        # CRITICAL: Only use observations that have successfully computed footprints
        # obs_dict contains only observations with successful footprint computation
        if not hasattr(self, 'obs_dict') or len(self.obs_dict) == 0:
            logger.warning("No footprints computed. Skipping train/test creation.")
            return

        # Filter obs_df to only include indices in obs_dict
        obs_indices_with_footprints = list(self.obs_dict.keys())
        df_full = self.obs_df.loc[self.obs_df.index.isin(obs_indices_with_footprints)].copy()

        logger.info(f"Total observations in domain: {len(self.obs_df)}")
        logger.info(f"Observations with successful footprints: {len(df_full)}")
        logger.info(f"Excluded (no footprints): {len(self.obs_df) - len(df_full)}")

        # Compute bkg_ref for each observation with footprints
        logger.info(f"Adding bkg_ref from precomputed trajectories...")
        bkg_refs = []
        for idx in df_full.index:
            if idx < len(self.trajectories):
                val = self.trajectories[idx]
                ref_list = val["ref_list"] if val is not None else []
            else:
                ref_list = []
            # Convert numpy types to Python native types for CSV serialization
            if ref_list:
                ref_list = [[item[0], float(item[1]), item[2], float(item[3]), float(item[4])] for item in ref_list]
            bkg_refs.append(ref_list)

        df_full["bkg_ref"] = bkg_refs

        # Sort by time to ensure sequential train/test split spans all dates
        df_full = df_full.sort_values("time").reset_index(drop=True)
        logger.info(f"✓ Added bkg_ref to all observations and sorted by time")

        # Split into train/test (80-20 split, sequential)
        n_train = int(0.8 * len(df_full))
        df_train = df_full.iloc[:n_train].reset_index(drop=True)
        df_test = df_full.iloc[n_train:].reset_index(drop=True)

        # Log train info (check if not empty to avoid .min()/.max() on empty series)
        if len(df_train) > 0:
            logger.info(f"  Train: {len(df_train)} observations ({df_train['time'].min().date()} to {df_train['time'].max().date()})")
        else:
            logger.warning(f"  Train: 0 observations (EMPTY - no train data!)")

        # Log test info
        if len(df_test) > 0:
            logger.info(f"  Test: {len(df_test)} observations ({df_test['time'].min().date()} to {df_test['time'].max().date()})")
        else:
            logger.warning(f"  Test: 0 observations (EMPTY - no test data!)")

        # Save train/test to config paths
        os.makedirs(os.path.dirname(config.tropomi_train_file), exist_ok=True)
        os.makedirs(os.path.dirname(config.tropomi_test_file), exist_ok=True)

        df_train.to_csv(config.tropomi_train_file, index=False)
        df_test.to_csv(config.tropomi_test_file, index=False)

        logger.info(f"✓ Saved {len(df_train)} training observations to {config.tropomi_train_file}")
        logger.info(f"✓ Saved {len(df_test)} test observations to {config.tropomi_test_file}")

    def find_obs(self, start_time, end_time, polygon_boundary, df=None):
        if df is None:
            logger.info(f"Loading {self.file}")
            df = pd.read_csv(self.file)
            df['lons'] = df['lon_str'].apply(lambda x: [float(val) for val in x.split("|")])
            df['lats'] = df['lat_str'].apply(lambda x: [float(val) for val in x.split("|")])
            df = df.drop(['lon_str', 'lat_str'], axis=1)
            df['geometry'] = df[['lons', 'lats']].apply(lambda x: Polygon(zip(x.iloc[0]+[x.iloc[0][0]], x.iloc[1]+[x.iloc[1][0]])), axis=1)
            logger.info(f"✓ Loaded {len(df)} observations from CSV")
            df['delta_time'] = df['delta_time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
            df['actual_time'] = df['actual_time'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
            df['time'] = df['actual_time'].apply(lambda x:x.round('60min').to_pydatetime())
            df = gpd.GeoDataFrame(df, geometry="geometry")
        
        # dk = df[(self.start_time <= df['actual_time']) & (df['actual_time'] <= self.end_time) & (self.upwind_lons[0] <= df['lon']) & (df['lon'] <= self.upwind_lons[-1]) & (self.upwind_lats[0] <= df['lat']) & (df['lat'] <= self.upwind_lats[-1])].reset_index(drop=True)
        dk = df[(start_time <= df['actual_time']) & (df['actual_time'] <= end_time) & (df.geometry.within(polygon_boundary))].reset_index(drop=True)
        return dk

    def compute_subpixels(self, obs_df, lats, lons):
        logger.debug(f"Grid size: {lats.shape[0]} × {lons.shape[0]} = {lats.shape[0] * lons.shape[0]} cells")

        subpixel_grid = pd.DataFrame(np.vstack((np.repeat(lats, lons.shape[0]), np.tile(lons, lats.shape[0]))).T, columns=['grid_lat', 'grid_lon'])
        logger.debug(f"Subpixel grid created: {subpixel_grid.shape[0]} cells")

        subpixel_grid = gpd.GeoDataFrame(geometry=[Point(subpixel_grid['grid_lon'][idx], subpixel_grid['grid_lat'][idx]) for idx in range(subpixel_grid.shape[0])])
        logger.debug(f"Subpixel grid as GeoDataFrame: {subpixel_grid.shape[0]} geometries")

        logger.debug(f"Observations before join: {obs_df.shape[0]} rows")
        logger.debug(f"Sample observation geometry: {obs_df['geometry'].iloc[0]}")

        obs_df['gp_geometry'] = obs_df['geometry']
        subpixels_df = gpd.sjoin(obs_df, subpixel_grid, how='right')
        logger.debug(f"After spatial join: {subpixels_df.shape[0]} rows")
        logger.debug(f"NaN rows before dropna: {subpixels_df.isna().sum().sum()}")

        subpixels_df = subpixels_df.dropna().reset_index(drop=True)
        logger.info(f"After dropna: {subpixels_df.shape[0]} rows")

        if subpixels_df.shape[0] > 0:
            subpixels_df['centroid_lon'] = [val.centroid.x for val in subpixels_df['geometry']]
            subpixels_df['centroid_lat'] = [val.centroid.y for val in subpixels_df['geometry']]
            subpixels_df = subpixels_df.sort_values(by="index_left")
        else:
            logger.error("ERROR: No subpixels after dropna()! This means spatial join found no overlaps.")
            logger.error(f"Check: Do observation geometries overlap with grid? Sample obs bounds: {obs_df['geometry'].iloc[0].bounds if len(obs_df) > 0 else 'N/A'}")

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
        refs = [val["ref_list"] if val is not None else None for val in self.trajectories]
        norm_dist = [val["norm_dist"] if val is not None else None for val in self.trajectories]
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
        logger.info("="*60)
        logger.info("Computing FootNet footprints")
        logger.info("="*60)

        # Gathering meteorology
        timelist = list(set(self.obs_df["time"]))
        logger.info(f"Loading meteorology for {len(timelist)} unique times...")
        start_met = time.time()
        self.input_met_foot = ColumnMeteorology(timelist, self.lons, self.lats, self.config.trimsize, self.config.hr3lat_full, self.config.hr3lon_full, self.config.HRRR_DIR, backhours=[0, 6, 12, 18, 24])
        logger.info(f"✓ Meteorology loaded in {time.time()-start_met:.1f}s")

        logger.info(f"Loading FootNet model from {self.config.model_path}...")
        model = ColumnFootNet(model_path=self.config.model_path)
        logger.info("✓ Model loaded")

        logger.info(f"Computing footprints for {self.subpixels_df.shape[0]} subpixels ({len(self.obs_df)} observations)...")
        self.obs_dict = {}
        grouped = self.subpixels_df.groupby("index_left")
        n_total = len(grouped)
        skipped_observations = []

        for obs_num, (idx, data) in enumerate(tqdm(grouped), 1):
            logger.debug(f"[{obs_num}/{n_total}] Processing observation {idx}")
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
                try:
                    # Check if meteorology is available for this timestamp
                    tstamp_key = datetime.datetime.strftime(tstamp, "%Y%m%d%H")

                    # Verify meteorology exists and is not None
                    if tstamp_key not in self.input_met_foot.input_met_dict:
                        raise ValueError(f"Meteorology key {tstamp_key} not found in input_met_dict")
                    if self.input_met_foot.input_met_dict[tstamp_key] is None:
                        raise ValueError(f"Missing HRRR meteorology data for {tstamp_key}")
                    if tstamp_key not in self.input_met_foot.processed_met_dict:
                        raise ValueError(f"Processed meteorology key {tstamp_key} not found")
                    if self.input_met_foot.processed_met_dict[tstamp_key] is None:
                        raise ValueError(f"Missing processed meteorology data for {tstamp_key}")

                    receptors = [[data['time'][jdx].to_pydatetime(), float(data['centroid_lon'][jdx]), float(data['centroid_lat'][jdx])] for jdx in range(data.shape[0])]
                    # Gather meteorology

                    # Compute footprints
                    foots, reference_indices, reference_timestamps, reference_rlons, reference_rlats, reference_foot_hours = model.run_inference(receptors, self.input_met_foot, maximum_domain_trajectory=self.config.maximum_domain_trajectory)

                    # print("Row:", row)
                    self.obs_dict[int(idx)]['foot'] = np.mean(foots, axis=0)
                    self.obs_dict[int(idx)]["avg_transport_hours"] = np.mean(reference_foot_hours)
                    self.write_footprint_file(filename, self.obs_dict[int(idx)])
                except Exception as e:
                    logger.warning(f"Skipping observation {int(idx)} (time: {self.obs_dict[int(idx)]['time']}) - missing meteorology or computation error: {e}")
                    skipped_observations.append(int(idx))
                    # Remove from obs_dict since we couldn't compute footprint
                    if int(idx) in self.obs_dict:
                        del self.obs_dict[int(idx)]

        # Log summary
        logger.info(f"✓ Footprint computation complete")
        logger.info(f"  Successfully computed: {len(self.obs_dict)} observations")
        if skipped_observations:
            logger.info(f"  Skipped (missing HRRR): {len(skipped_observations)} observations")
            logger.info(f"  Skipped observation indices: {skipped_observations[:10]}{'...' if len(skipped_observations) > 10 else ''}")
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
        rlon[:] = obs_dict["lon"]
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
        plt.close()  # Close figure instead of showing (blocking)



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
        self.ems_buffer_days = cfs["ems_buffer_days"]
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
        self.tropomi_bkg_file = cfs["tropomi_bkg_file"]
        self.tropomi_train_file = cfs["tropomi_train_file"]
        self.tropomi_test_file = cfs["tropomi_test_file"]
        
if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="PATH to YAML config file")
        parser.add_argument("--start_time", required=True, help="Inversion start date")
        parser.add_argument("--end_time", required=True, help="Inversion end date")
        args = parser.parse_args()

        logger.info(f"Loading config from: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg["start_time"] = args.start_time
            cfg["end_time"] = args.end_time

        config = TROPOMI_config(cfg)
        logger.info("✓ Config loaded successfully")

        tropomi = getTROPOMI(config)
        logger.info(f"Subpixels dataframe: {tropomi.subpixels_df.shape[0]} rows")

        logger.info("Plotting domain visualization...")
        tropomi.plot_domains()
        logger.info("✓ domain.png saved")

        tropomi.compute_footprints()

        # Create train/test split using ONLY observations with successful footprints
        tropomi.create_and_save_train_test_data_with_footprints(config)

        logger.info("="*60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)

    except Exception as e:
        logger.error("="*60)
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        logger.error("="*60)
        raise
    