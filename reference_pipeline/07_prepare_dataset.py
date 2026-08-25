# ---------------------------------------------------------------------------
# REFERENCE ONLY -- not runnable from this repository.
#
# This script belongs to the upstream pipeline that produced the analysis
# dataset. It requires a Google Street View API key and/or the panorama store
# held on the lab server, neither of which is redistributable. It is included
# to document how the published features were derived, not to be re-executed.
#
# Set PERSONALITY_SVI_ROOT and SVI_IMAGE_DIR if you adapt it to your own data.
# ---------------------------------------------------------------------------

# -*- coding: utf-8 -*-
# NOTE: keep the first character of the file as the '#', not a hidden space

import os
import sys
from pathlib import Path

# --- Make local modules resolvable no matter where you run this file ---
# Repo root: .../250529_personality_svi
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Package dir: .../250529_personality_svi/personality_svi
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# --- Standard libs and deps ---
import polars as pl
import geopandas as gpd
import pyreadstat
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse
from tqdm import tqdm

# --- Local modules (short, relative-style) ---
from personality_svi.data_processing_01.big_five_analysis import calculate_personality_scores_vectorized
from personality_svi.data_processing_01.census_data import get_zipcode_population_data_census
from personality_svi import config

from personality_svi.config import (
    PROJ_ROOT,
    RAW_DATA_DIR as CFG_RAW_DATA_DIR,
    PROCESSED_DATA_DIR as CFG_PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR as CFG_EXTERNAL_DATA_DIR,
    ZIP_SHAPEFILE_PATH,
    SURVEY_SAV_PATH,
)

# Configuration (use config paths)
WORKSPACE_DIR = PROJ_ROOT
RAW_DATA_DIR = CFG_RAW_DATA_DIR
PROCESSED_DATA_DIR = CFG_PROCESSED_DATA_DIR
SEGMENTATION_DIR = PROCESSED_DATA_DIR / "segmentation"
DETECTION_DIR = PROCESSED_DATA_DIR / "detection"
CLASSIFICATION_DIR = PROCESSED_DATA_DIR / "classification"
CHECKPOINT_DIR = PROCESSED_DATA_DIR / "checkpoints"
FINAL_INDIVIDUAL_DATA_PATH = PROCESSED_DATA_DIR / "model/final_dataset_individual.parquet"
FINAL_ZIPCODE_DATA_PATH = PROCESSED_DATA_DIR / "model/final_dataset_zipcode.parquet"
FINAL_ZIPCODE_SPATIAL_DATA_PATH = PROCESSED_DATA_DIR / "model/final_dataset_zipcode_spatial.parquet"
FINAL_ZIPCODE_TEMPORAL_DATA_PATH = PROCESSED_DATA_DIR / "model/final_dataset_zipcode_temporal.parquet"
FINAL_ZIPCODE_AGGREGATED_PATH = PROCESSED_DATA_DIR / "model/final_dataset_zipcode_aggregated.parquet"
FINAL_ZIPCODE_SPATIAL_AGGREGATED_PATH = PROCESSED_DATA_DIR / "model/final_dataset_zipcode_spatial_aggregated.parquet"
FINAL_ZIPCODE_TEMPORAL_AGGREGATED_PATH = PROCESSED_DATA_DIR / "model/final_dataset_zipcode_temporal_aggregated.parquet"
INTERIM_DIR = PROJ_ROOT / "data/interim"
SURVEY_PARQUET_PATH = INTERIM_DIR / "survey_data.parquet"

# Model parameters
FLAG_DETECTION_THRESHOLD = 0.75  # Confidence threshold for American flag detection

# Do not create directories here; assume paths exist per config

def load_or_create_checkpoint(path: Path, create_func, *args, **kwargs):
    """Helper function to load checkpoint if exists or create and save if not"""
    if path.exists():
        print(f"Loading checkpoint from {path}")
        return pl.read_parquet(path)
    
    print(f"Creating new data for {path}")
    df = create_func(*args, **kwargs)

    # Ensure parent directory exists before saving
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoint to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df

def load_survey_data():
    """Load survey from interim/survey_data.parquet if present; otherwise build from SAV once."""
    # Case 1: preferred location already exists -> just read and return
    if SURVEY_PARQUET_PATH.exists():
        print(f"Loading survey parquet from {SURVEY_PARQUET_PATH}")
        return pl.read_parquet(SURVEY_PARQUET_PATH)

    # # Case 2: legacy checkpoint exists -> copy to interim (no overwrite), then use it
    # if LEGACY_SURVEY_CHECKPOINT.exists():
    #     print(f"Found legacy survey checkpoint at {LEGACY_SURVEY_CHECKPOINT}")
    #     INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    #     # copy only if interim doesn't already have it (we already checked it doesn't)
    #     import shutil
    #     shutil.copy2(LEGACY_SURVEY_CHECKPOINT, SURVEY_PARQUET_PATH)
    #     print(f"Copied to {SURVEY_PARQUET_PATH}")
    #     return pl.read_parquet(SURVEY_PARQUET_PATH)

    # Case 3: build from SAV and write to interim once
    print("survey_data.parquet not found; loading SAV and materializing interim parquet...")
    df_pandas, _meta = pyreadstat.read_sav(str(SURVEY_SAV_PATH))
    df = pl.from_pandas(df_pandas)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(SURVEY_PARQUET_PATH)
    print(f"Wrote survey parquet to {SURVEY_PARQUET_PATH}")
    return df

def load_zipcode_shapes():
    """Load and process zipcode shapefile"""
    gdf = gpd.read_file(str(ZIP_SHAPEFILE_PATH))
    return gdf

def process_census_data(gdf_zipcodes, year=2020):
    """Process US Census population and demographic data for zipcodes.

    Uses Census Bureau data instead of WorldPop for reliable, authoritative demographic data.
    """
    year_checkpoint_path = CHECKPOINT_DIR / f"census_data_{year}.parquet"
    def _create_census_data():
        print(f"Fetching US Census population and demographic data for year {year}...")
        return get_zipcode_population_data_census(
            gdf_zipcodes=gdf_zipcodes,
            year=year,
            cache_path=year_checkpoint_path,
            batch_size=10  # Not used but kept for compatibility
        )
    return load_or_create_checkpoint(year_checkpoint_path, _create_census_data)

def process_gsv_metadata():
    """Process Google Street View metadata from all cities"""
    checkpoint_path = CHECKPOINT_DIR / "gsv_metadata.parquet"
    
    def _create_gsv_metadata():
        city_dfs = []
        for city_dir in RAW_DATA_DIR.glob("*"):
            if not city_dir.is_dir():
                continue
                
            metadata_file = city_dir / "gsv_pids.csv"
            if not metadata_file.exists():
                continue
                
            df = pl.read_csv(metadata_file)
            df = df.with_columns(pl.lit(city_dir.name).alias('city'))
            city_dfs.append(df)
        
        if not city_dfs:
            raise FileNotFoundError("No GSV metadata files found")
            
        return pl.concat(city_dfs)
    
    return load_or_create_checkpoint(checkpoint_path, _create_gsv_metadata)

def process_zipcode_year(zipcode, year, pl_metrics, zipcode_area_dict):
    """Process a single zipcode for spatial metrics using polars for faster filtering"""
    try:
        # Get zipcode area from pre-computed dictionary
        if zipcode not in zipcode_area_dict:
            return (zipcode, year), None
            
        zipcode_area = zipcode_area_dict[zipcode]
        
        # Filter points for this zipcode using polars
        zip_points = pl_metrics.filter(pl.col('zipcode') == zipcode)
        
        # Skip if no points
        if len(zip_points) == 0:
            return (zipcode, year), None
        
        # Number of points
        point_count = len(zip_points)
        
        # Density (points per square unit)
        density = point_count / zipcode_area
        
        # Return metrics (simplified for speed)
        return (zipcode, year), {
            'point_count': point_count,
            'density': density
        }
    
    except Exception as e:
        print(f"Error processing zipcode {zipcode}: {str(e)}")
        return (zipcode, year), None

def process_zipcode_year_with_grid_coverage(zipcode, year, pl_metrics, zipcode_area_dict, gdf_zipcode_shapes, gdf_metrics_proj, grid_size=100):
    """
    Process a single zipcode for spatial metrics using polars for faster filtering,
    with added grid-based coverage calculation using 100m grid cells.
    
    Parameters:
    -----------
    zipcode : str
        The zipcode to process
    year : int or None
        The year to filter by, or None to not filter by year (ignored in spatial-only matching)
    pl_metrics : polars.DataFrame
        Metrics dataframe with zipcode and panoid columns
    zipcode_area_dict : dict
        Dictionary mapping zipcode to area in square meters
    gdf_zipcode_shapes : geopandas.GeoDataFrame
        GeoDataFrame containing zipcode boundaries
    gdf_metrics_proj : geopandas.GeoDataFrame
        GeoDataFrame with GSV point locations
    grid_size : int, default=100
        Size of grid cells in meters
        
    Returns:
    --------
    tuple of (key, result_dict)
        key is (zipcode, year) and result_dict contains metrics
    """
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import box
        
        # Get zipcode area from pre-computed dictionary
        if zipcode not in zipcode_area_dict:
            return (zipcode, year), None
            
        zipcode_area = zipcode_area_dict[zipcode]
        
        # Filter points for this zipcode using polars
        zip_points = pl_metrics.filter(pl.col('zipcode') == zipcode)
        
        # Skip if no points
        if len(zip_points) == 0:
            return (zipcode, year), None
        
        # Number of points
        point_count = len(zip_points)
        
        # Density (points per square unit)
        density = point_count / zipcode_area
        
        # Create the grid-based coverage calculation
        try:
            # Get the zipcode polygon
            zipcode_polygon = gdf_zipcode_shapes[gdf_zipcode_shapes['ZCTA5CE10'] == zipcode]
            
            if len(zipcode_polygon) == 0:
                print(f"Warning: No polygon found for zipcode {zipcode}")
                return (zipcode, year), {
                    'point_count': point_count,
                    'density': density,
                    'grid_coverage': None
                }
            
            # Get the bounds of the zipcode polygon
            bounds = zipcode_polygon.geometry.iloc[0].bounds
            xmin, ymin, xmax, ymax = bounds
            
            # Grid cell size in meters (100m x 100m)
            cell_size = grid_size
            
            # Create grid cells
            grid_cells = []
            print(f"  Generating {cell_size}m grid for zipcode {zipcode}...")
            
            # More efficient vectorized grid creation
            # Create coordinate arrays for grid cells
            x_coords = np.arange(xmin, xmax + cell_size, cell_size)
            y_coords = np.arange(ymin, ymax + cell_size, cell_size)
            
            # Get the zipcode geometry for intersection check
            zipcode_geom = zipcode_polygon.geometry.iloc[0]
            
            # Create grid cells vectorized, when possible
            try:
                # For simple polygons, we can use vectorized operations
                if len(x_coords) * len(y_coords) < 10000:  # Only for reasonably sized grids
                    # Create meshgrid for all combinations
                    xx, yy = np.meshgrid(x_coords[:-1], y_coords[:-1])
                    
                    # Create all boxes at once
                    boxes = [box(x, y, x + cell_size, y + cell_size) 
                            for x, y in zip(xx.flatten(), yy.flatten())]
                    
                    # Filter boxes that intersect with the zipcode polygon
                    grid_cells = [b for b in boxes if b.intersects(zipcode_geom)]
                else:
                    # Fall back to loop for very large grids
                    for x0 in x_coords[:-1]:
                        for y0 in y_coords[:-1]:
                            # Create the cell
                            cell = box(x0, y0, x0 + cell_size, y0 + cell_size)
                            # Only include cells that intersect with the zipcode polygon
                            if cell.intersects(zipcode_geom):
                                grid_cells.append(cell)
            except Exception as e:
                print(f"  Falling back to standard grid creation due to error: {e}")
                # Fall back to the original loop implementation
                grid_cells = []
                for x0 in np.arange(xmin, xmax + cell_size, cell_size):
                    for y0 in np.arange(ymin, ymax + cell_size, cell_size):
                        # Create the cell
                        cell = box(x0, y0, x0 + cell_size, y0 + cell_size)
                        # Only include cells that intersect with the zipcode polygon
                        if cell.intersects(zipcode_polygon.geometry.iloc[0]):
                            grid_cells.append(cell)
            
            print(f"  Created {len(grid_cells)} grid cells for zipcode {zipcode}")
            
            # Create a GeoDataFrame from the grid cells
            grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs='EPSG:3857')
            
            # Filter the metrics for this zipcode (spatial-only matching)
            metrics_subset = gdf_metrics_proj[gdf_metrics_proj['zipcode'] == zipcode]
            
            # Use spatial join to efficiently find cells with points
            # This is much faster than iterating through each cell
            if len(metrics_subset) > 0:
                print(f"  Performing spatial join for {len(metrics_subset)} points in zipcode {zipcode}...")
                # Create a copy of metrics_subset as a GeoDataFrame with the same CRS
                metrics_gdf = gpd.GeoDataFrame(
                    {'geometry': metrics_subset.geometry},
                    crs=grid_gdf.crs
                )
                
                # Perform spatial join to find which grid cells have points
                joined = gpd.sjoin(grid_gdf, metrics_gdf, how='left', predicate='contains')
                
                # Count unique cells with at least one point
                cells_with_points = joined.dropna().index.nunique()
            else:
                cells_with_points = 0
            
            # Calculate coverage ratio
            total_cells = len(grid_cells)
            coverage_ratio = cells_with_points / total_cells if total_cells > 0 else 0
            
            print(f"  Zipcode {zipcode}: {cells_with_points}/{total_cells} cells have GSV points (coverage: {coverage_ratio:.2%})")
            
            return (zipcode, year), {
                'point_count': point_count,
                'density': density,
                'grid_cell_count': total_cells,
                'cells_with_points': cells_with_points,
                'grid_coverage': coverage_ratio
            }
            
        except Exception as e:
            print(f"Grid coverage calculation error for {zipcode}: {str(e)}")
            # Return basic metrics without grid coverage if the calculation fails
            return (zipcode, year), {
                'point_count': point_count,
                'density': density
            }
    
    except Exception as e:
        print(f"Error processing zipcode {zipcode}: {str(e)}")
        return (zipcode, year), None

def spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes, grid_size=100):
    """
    Perform spatial join between GSV points and zipcode polygons
    
    Parameters:
    -----------
    df_gsv : polars.DataFrame
        DataFrame containing GSV data
    gdf_zipcodes : geopandas.GeoDataFrame
        GeoDataFrame containing zipcode polygon shapes
    grid_size : int, default=100
        Size of grid cells in meters for coverage calculation
    """
    # Use grid-size specific checkpoint for reproducibility
    checkpoint_path = CHECKPOINT_DIR / f"gsv_with_zipcodes_grid{grid_size}m.parquet"
    legacy_checkpoint_path = CHECKPOINT_DIR / "gsv_with_zipcodes.parquet"

    # Backward compatibility: if legacy exists and grid-specific does not, duplicate it
    if not checkpoint_path.exists() and legacy_checkpoint_path.exists():
        print(f"Found legacy GSV zipcode join at {legacy_checkpoint_path}; copying to grid-specific file {checkpoint_path}")
        pl.read_parquet(legacy_checkpoint_path).write_parquet(checkpoint_path)
    
    def _create_spatial_join(df_gsv=df_gsv, gdf_zipcodes=gdf_zipcodes, grid_size=grid_size):
        # Convert Polars DataFrame to GeoDataFrame
        gdf_gsv = gpd.GeoDataFrame(
            df_gsv.to_pandas(),
            geometry=gpd.points_from_xy(df_gsv['lon'], df_gsv['lat']),
            crs="EPSG:4326"
        )
        
        # Ensure both GeoDataFrames have the same CRS
        if gdf_zipcodes.crs != gdf_gsv.crs:
            gdf_zipcodes = gdf_zipcodes.to_crs(gdf_gsv.crs)
        
        # Create a zipcode area lookup dictionary for later use
        # Convert to projected CRS for accurate area calculation
        gdf_zipcodes_proj = gdf_zipcodes.to_crs('EPSG:3857')
        zipcode_areas = {code: float(area) for code, area in 
                        zip(gdf_zipcodes_proj['ZCTA5CE10'], gdf_zipcodes_proj.geometry.area)}
        
        # Perform spatial join
        print("Performing spatial join...")
        gdf_joined = gpd.sjoin(gdf_gsv, gdf_zipcodes[['ZCTA5CE10', 'geometry']], 
                              how='left', predicate='within')
        
        # Already rename the zipcode column in the GeoDataFrame before making a copy
        gdf_joined = gdf_joined.rename(columns={'ZCTA5CE10': 'zipcode'})
        
        # Create a copy of the GeoDataFrame for spatial calculations
        gdf_metrics = gdf_joined.copy()
        
        # Filter only rows with valid zipcodes
        gdf_metrics = gdf_metrics[~gdf_metrics['zipcode'].isna()]
        
        # Convert back to Polars DataFrame (after making the copy for metrics)
        df_joined = pl.from_pandas(gdf_joined.drop(columns=['geometry', 'index_right']))
        
        # Add zipcode area column to the dataframe
        # Explicitly cast to float64 to ensure it's numeric for calculations
        df_joined = df_joined.with_columns(
            pl.col('zipcode').replace(zipcode_areas).cast(pl.Float64).alias('zipcode_area_m2')
        )
        
        # Calculate spatial metrics for each zipcode
        print("Calculating spatial metrics...")
        
        # Project points for accurate calculations
        gdf_metrics_proj = gdf_metrics.to_crs('EPSG:3857')
        
        # Convert to polars DataFrame for faster filtering operations
        # Extract only the columns we need for filtering and calculations
        pl_metrics = pl.from_pandas(
            gdf_metrics_proj[['zipcode', 'panoid']].reset_index(drop=True)
        )
        
        # Create empty dictionary to store metrics
        zipcode_metrics = {}
        
        # Get unique zipcodes
        zipcodes = pl_metrics['zipcode'].unique().to_list()
        
        # Process each zipcode SEQUENTIALLY with polars filtering
        print(f"Processing {len(zipcodes)} zipcodes sequentially...")
        
        # Use tqdm for progress tracking in sequential processing
        for i, zipcode in enumerate(tqdm(zipcodes, desc="Calculating metrics by zipcode")):
            try:
                # Use the function with grid coverage calculation, passing None for year
                key, result = process_zipcode_year_with_grid_coverage(
                    zipcode, None, pl_metrics, zipcode_areas, 
                    gdf_zipcodes_proj, gdf_metrics_proj, grid_size
                )
                
                if result is not None:
                    zipcode_metrics[key] = result
                
                # Print progress occasionally
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(zipcodes)} zipcodes")
            except Exception as e:
                print(f"Error processing zipcode {zipcode}: {str(e)}")
                # Continue processing other zipcodes even if one fails
                continue
        
        # Convert metrics to DataFrame
        metrics_rows = []
        for (zipcode, _), metrics in zipcode_metrics.items():
            if metrics is not None:
                metrics_dict = {
                    'zipcode': zipcode,
                    'gsv_point_count': metrics['point_count'],
                    'gsv_density': metrics['density']
                }
                
                # Add grid coverage metrics if they exist
                if 'grid_coverage' in metrics and metrics['grid_coverage'] is not None:
                    metrics_dict.update({
                        'grid_cell_count': metrics['grid_cell_count'],
                        'cells_with_points': metrics['cells_with_points'],
                        'grid_coverage': metrics['grid_coverage']
                    })
                
                metrics_rows.append(metrics_dict)
        
        df_metrics = pl.from_pandas(pd.DataFrame(metrics_rows))
        
        # Join metrics back to the original dataframe using just zipcode
        join_keys = ['zipcode']
            
        # Ensure metrics dataframe has the right schema for the join
        if len(metrics_rows) > 0:
            df_joined = df_joined.join(
                df_metrics,
                on=join_keys,
                how='left',
                coalesce=True
            )
        
        return df_joined
    
    return load_or_create_checkpoint(checkpoint_path, _create_spatial_join)

def process_segmentation_results():
    """Process segmentation results from all cities and batches"""
    checkpoint_path = CHECKPOINT_DIR / "segmentation_results.parquet"
    
    def _create_segmentation_results():
        results_dfs = []
        
        for city_dir in SEGMENTATION_DIR.glob("*"):
            if not city_dir.is_dir():
                continue
                
            for batch_dir in city_dir.glob("batch_*"):
                if not batch_dir.is_dir():
                    continue
                    
                pixel_ratios = batch_dir / "pixel_ratios.csv"
                label_counts = batch_dir / "label_counts.csv"
                
                if not pixel_ratios.exists() or not label_counts.exists():
                    continue
                    
                # Read with suffixes for all columns
                df_ratios = pl.read_csv(pixel_ratios)
                df_counts = pl.read_csv(label_counts)
                
                # Add suffixes to all columns except the joining key
                ratio_cols = [col for col in df_ratios.columns if col != 'filename_key']
                count_cols = [col for col in df_counts.columns if col != 'filename_key']
                
                df_ratios = df_ratios.rename({col: f"{col}_ratios" for col in ratio_cols})
                df_counts = df_counts.rename({col: f"{col}_counts" for col in count_cols})
                
                # Merge ratios and counts
                df = df_ratios.join(df_counts, on='filename_key')
                df = df.with_columns([
                    pl.lit(city_dir.name).alias('city'),
                    pl.lit(batch_dir.name).alias('batch')
                ])
                print(f"Processing {city_dir.name}/{batch_dir.name}: {df.shape}")
                results_dfs.append(df)
        
        if not results_dfs:
            raise FileNotFoundError("No segmentation result files found")
            
        # Concatenate with align=True to handle different columns
        print("Concatenating all results...")
        df_concat = pl.concat(results_dfs, how="diagonal")
        
        # Fill null values with 0
        print("Filling null values with 0...")
        numeric_cols = [col for col in df_concat.columns 
                       if col not in ['filename_key', 'city', 'batch'] 
                       and df_concat[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        df_concat = df_concat.with_columns([
            pl.col(col).fill_null(0) for col in numeric_cols
        ])
        
        return df_concat
    
    return load_or_create_checkpoint(checkpoint_path, _create_segmentation_results)

def calculate_visual_complexity(df):
    """Calculate Shannon index for each image based on pixel ratios"""
    # Get ratio columns
    ratio_cols = [col for col in df.columns if col.endswith('_ratios')]
    
    # Convert to numpy for faster computation
    ratios_array = df[ratio_cols].to_numpy()
    
    # Calculate Shannon index for each row
    complexity = np.array([shannon_index(row) for row in ratios_array])
    
    return pl.Series(complexity)

def shannon_index(ratios):
    # Remove zero values and normalize
    nonzero_ratios = ratios[ratios > 0]
    if len(nonzero_ratios) == 0:
        return 0
    normalized = nonzero_ratios / nonzero_ratios.sum()
    # Calculate Shannon index
    return -(normalized * np.log(normalized)).sum()

def process_detection_results():
    """Process American flag detection results from all cities and batches"""
    checkpoint_path = CHECKPOINT_DIR / "detection_results.parquet"
    
    def _create_detection_results():
        results_dfs = []
        
        for city_dir in DETECTION_DIR.glob("*"):
            if not city_dir.is_dir():
                continue
                
            for batch_dir in city_dir.glob("batch_*"):
                if not batch_dir.is_dir():
                    continue
                    
                detection_summary = batch_dir / "detection_summary.csv"
                
                if not detection_summary.exists():
                    continue
                    
                # Read detection summary
                df = pl.read_csv(detection_summary)
                
                # Filter by confidence threshold
                df = df.filter(pl.col('logit') >= FLAG_DETECTION_THRESHOLD)
                
                # Group by filename_key to count flags per image
                df = df.group_by('filename_key').agg([
                    pl.count('logit').alias('flag_count'),
                    pl.mean('logit').alias('flag_confidence_mean'),
                    pl.max('logit').alias('flag_confidence_max')
                ])
                
                df = df.with_columns([
                    pl.lit(city_dir.name).alias('city'),
                    pl.lit(batch_dir.name).alias('batch')
                ])
                
                print(f"Processing {city_dir.name}/{batch_dir.name}: {df.shape}")
                results_dfs.append(df)
        
        if not results_dfs:
            raise FileNotFoundError("No detection result files found")
            
        # Concatenate all results
        print("Concatenating all results...")
        df_concat = pl.concat(results_dfs, how="diagonal")
        
        # Fill null values with 0 (for images without flags above threshold)
        print("Filling null values with 0...")
        numeric_cols = ['flag_count', 'flag_confidence_mean', 'flag_confidence_max']
        df_concat = df_concat.with_columns([
            pl.col(col).fill_null(0) for col in numeric_cols
        ])
        
        return df_concat
    
    return load_or_create_checkpoint(checkpoint_path, _create_detection_results)

def process_classification_results():
    """Process classification results from all cities and batches"""
    checkpoint_path = CHECKPOINT_DIR / "classification_results.parquet"
    
    def _create_classification_results():
        results_dfs = []
        
        for city_dir in CLASSIFICATION_DIR.glob("*"):
            if not city_dir.is_dir():
                continue
                
            for batch_dir in city_dir.glob("batch_*"):
                if not batch_dir.is_dir():
                    continue
                    
                classification_summary = batch_dir / "results.csv"
                
                if not classification_summary.exists():
                    continue
                    
                try: 
                    # Read classification summary
                    df = pl.read_csv(classification_summary)

                except:
                   continue 
                
                # Get the environment type for each image
                # Assuming the file has columns for filename_key and environment_type
                df = df.with_columns([
                    pl.lit(city_dir.name).alias('city'),
                    pl.lit(batch_dir.name).alias('batch')
                ])
                
                print(f"Processing {city_dir.name}/{batch_dir.name}: {df.shape}")
                results_dfs.append(df)
        
        if not results_dfs:
            raise FileNotFoundError("No classification result files found")
            
        # Concatenate all results
        print("Concatenating all classification results...")
        df_concat = pl.concat(results_dfs, how="diagonal")
        
        return df_concat
    
    return load_or_create_checkpoint(checkpoint_path, _create_classification_results)

def aggregate_by_zipcode(df_gsv_with_zipcode, df_segments, df_detections=None, df_classifications=None, grid_size=100):
    """
    Aggregate segmentation and detection results by zipcode with spatial matching
    """
    # Parameterized checkpoints
    checkpoint_path = CHECKPOINT_DIR / f"segment_stats_by_zipcode_grid{grid_size}m.parquet"
    spatial_checkpoint_path = CHECKPOINT_DIR / f"segment_stats_by_zipcode_spatial_grid{grid_size}m.parquet"
    # Legacy (non-parameter) filenames for backward compatibility
    legacy_checkpoint_path = CHECKPOINT_DIR / "segment_stats_by_zipcode.parquet"
    legacy_spatial_checkpoint_path = CHECKPOINT_DIR / "segment_stats_by_zipcode_spatial.parquet"

    # If parameterized file missing but legacy exists, copy it forward
    if not checkpoint_path.exists() and legacy_checkpoint_path.exists():
        print(f"Found legacy aggregation checkpoint {legacy_checkpoint_path}; copying to {checkpoint_path}")
        pl.read_parquet(legacy_checkpoint_path).write_parquet(checkpoint_path)
    if not spatial_checkpoint_path.exists() and legacy_spatial_checkpoint_path.exists():
        print(f"Found legacy spatial aggregation checkpoint {legacy_spatial_checkpoint_path}; copying to {spatial_checkpoint_path}")
        pl.read_parquet(legacy_spatial_checkpoint_path).write_parquet(spatial_checkpoint_path)
    
    def _create_aggregation():
        # Load survey data to get years
        survey_data = load_survey_data()
        
        # Merge GSV metadata with segmentation results
        print("  Merging GSV metadata with segmentation results...")
        df_merged = df_gsv_with_zipcode.join(df_segments, left_on='panoid', right_on='filename_key', how='left', coalesce=True)
        
        # Merge with detection results if available
        if df_detections is not None:
            print("  Merging with flag detection results...")
            df_merged = df_merged.join(
                df_detections.select([
                    pl.col('filename_key'),
                    pl.col('flag_count'),
                    pl.col('flag_confidence_mean'),
                    pl.col('flag_confidence_max')
                ]),
                left_on='panoid',
                right_on='filename_key',
                how='left',
                coalesce=True
            )
            # Fill missing values with 0 (for images without flags)
            df_merged = df_merged.with_columns([
                pl.col('flag_count').fill_null(0),
                pl.col('flag_confidence_mean').fill_null(0),
                pl.col('flag_confidence_max').fill_null(0)
            ])
            
            # Calculate flag density per image (flags per square km)
            # First make sure we have the zipcode area column
            if 'zipcode_area_m2' in df_merged.columns:
                # Ensure zipcode_area_m2 is properly cast as numeric
                df_merged = df_merged.with_columns([
                    pl.col('zipcode_area_m2').cast(pl.Float64),
                    pl.col('flag_count').cast(pl.Float64)
                ])
                # Calculate density per square km
                df_merged = df_merged.with_columns(
                    (pl.col('flag_count') / (pl.col('zipcode_area_m2') / 1_000_000)).alias('flag_density_per_km2')
                )
            else:
                print("Warning: zipcode_area_m2 column not found. Flag density calculation skipped.")
        
        # Merge with classification results if available
        if df_classifications is not None:
            print("  Merging with classification results...")
            df_merged = df_merged.join(
                df_classifications.select([
                    pl.col('filename_key'), 
                    pl.col('environment_type')
                ]),
                left_on='panoid',
                right_on='filename_key',
                how='left',
                coalesce=True
            )
            
            # Filter out indoor images
            if 'environment_type' in df_merged.columns:
                print("  Filtering out indoor images...")
                pre_filter_count = len(df_merged)
                df_merged = df_merged.filter(pl.col('environment_type') != 'indoor')
                post_filter_count = len(df_merged)
                print(f"  Removed {pre_filter_count - post_filter_count} indoor images. {post_filter_count} outdoor images remaining.")
        
        # Calculate visual complexity for each image
        print("  Calculating visual complexity...")
        df_merged = df_merged.with_columns(
            calculate_visual_complexity(df_merged).alias('visual_complexity')
        )
        
        # Filter out rows with visual complexity = 0
        print("  Filtering out zero-complexity images...")
        pre_filter_complexity = len(df_merged)
        df_merged = df_merged.filter(pl.col('visual_complexity') > 1)
        post_filter_complexity = len(df_merged)
        print(f"  Removed {pre_filter_complexity - post_filter_complexity} low complexity images. {post_filter_complexity} images remaining.")
        
        # Get numeric columns for aggregation
        numeric_cols = [col for col in df_segments.columns 
                       if df_segments[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                       and col != 'filename_key']
        numeric_cols.append('visual_complexity')
        
        # Add detection metrics if available
        if df_detections is not None:
            numeric_cols.extend(['flag_count', 'flag_confidence_mean', 'flag_confidence_max'])
            if 'flag_density_per_km2' in df_merged.columns:
                numeric_cols.append('flag_density_per_km2')
        
        # Add the new spatial metrics columns if they exist
        spatial_metrics = ['gsv_point_count', 'gsv_density', 'zipcode_area_m2', 
                          'grid_cell_count', 'cells_with_points', 'grid_coverage']
        for metric in spatial_metrics:
            if metric in df_merged.columns:
                if metric not in numeric_cols:
                    numeric_cols.append(metric)
        
        # Focus on spatial-only matching
        print("Creating spatial-only matching dataset...")
        
        # Standard aggregation without year matching
        agg_exprs = [
            pl.col('panoid').count().alias('image_count'),
            # Take the most common city for each zipcode
            pl.col('city').mode().first().alias('city')
        ]
        
        # Add mean expressions
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean") 
            for col in numeric_cols
        ])
        
        # Add std expressions
        agg_exprs.extend([
            pl.col(col).std(ddof=1).cast(pl.Float64).alias(f"{col}_std")
            for col in numeric_cols
        ])
        
        df_spatial_result = df_merged.group_by('zipcode').agg(agg_exprs)
        print(f"  Created spatial-only dataset with {len(df_spatial_result)} zipcodes")
        
        # Save the spatial-only matching result
        df_spatial_result.write_parquet(spatial_checkpoint_path)
        print(f"  Saved spatial-only matching result to {spatial_checkpoint_path}")
        
        # Return the spatial dataset
        return df_spatial_result
    
    return load_or_create_checkpoint(checkpoint_path, _create_aggregation)

def create_final_dataset(df_survey, df_segment_stats, grid_size=100, censuspop_year=2020):
    """Create final datasets at both individual and zipcode levels with spatial matching"""
    print("Creating final datasets...")
    
    # Calculate Big Five personality traits with human-readable indicators
    print("  Calculating Big Five traits with human-readable indicators...")
    checkpoint_path = CHECKPOINT_DIR / "big_five_traits_with_prefixed_indicators.parquet"
    
    def _create_big_five():
        # This now includes prefixed human-readable indicators (ind_*)
        return pl.from_pandas(calculate_personality_scores_vectorized(df_survey.to_pandas()))
    
    df_big_five = load_or_create_checkpoint(checkpoint_path, _create_big_five)

    # --- AUDIT: Big Five calculation ---
    opn_valid = df_big_five['openness'].is_not_null().sum() if 'openness' in df_big_five.columns else 0
    ext_valid = df_big_five['extraversion'].is_not_null().sum() if 'extraversion' in df_big_five.columns else 0
    print(f"  [audit:big_five] rows={len(df_big_five):,}  openness_valid={opn_valid:,} ({opn_valid/max(len(df_big_five),1):.1%})  extraversion_valid={ext_valid:,}")

    # Check for human-readable indicators in the results
    indicator_columns = [col for col in df_big_five.columns
                        if col.startswith('ind_')]
    if indicator_columns:
        print(f"  Found {len(indicator_columns)} prefixed human-readable personality indicators (ind_*)")

    # Add Big Five traits to survey data
    df_combined = df_survey.hstack(df_big_five)
    print(f"  [audit:combined] rows={len(df_combined):,}  (survey + Big Five hstacked)")

    # Merge with street view indicators
    print("  Merging with street view indicators...")
    
    # Load spatial segment stats
    # Accept any grid size; prefer grid-specific spatial stats if multiple exist
    spatial_checkpoint_path = CHECKPOINT_DIR / f"segment_stats_by_zipcode_spatial_grid{100}m.parquet"  # default fallback
    # Try to find a grid-size specific file that matches common sizes (descending preference: user-provided size first)
    candidate = CHECKPOINT_DIR / f"segment_stats_by_zipcode_spatial_grid{grid_size}m.parquet"
    if candidate.exists():
        spatial_checkpoint_path = candidate
    else:
        # Fallback to legacy name
        legacy_candidate = CHECKPOINT_DIR / "segment_stats_by_zipcode_spatial.parquet"
        if legacy_candidate.exists():
            spatial_checkpoint_path = legacy_candidate
    
    # Load spatial stats
    if spatial_checkpoint_path.exists():
        print("  Loading spatial segment stats...")
        df_segment_stats_spatial = pl.read_parquet(spatial_checkpoint_path)
    else:
        # If no specific spatial stats file, use the provided segment stats
        df_segment_stats_spatial = df_segment_stats
    
    # --- AUDIT: before inner join ---
    n_pre_join = len(df_combined)
    print(f"  [audit:pre_join] rows={n_pre_join:,}  gsv_zips={df_segment_stats_spatial['zipcode'].n_unique()}")

    # Create individual dataset with spatial matching
    print("  Using spatial-only matching for individual dataset...")
    df_individual = df_combined.join(
        df_segment_stats_spatial,
        left_on='now_zip',
        right_on='zipcode',
        how='inner',  # Only keep rows with both personality and street view data
        coalesce=True
    )

    # --- AUDIT: after inner join ---
    n_post_join = len(df_individual)
    print(f"  [audit:after_join] rows={n_post_join:,}  lost={n_pre_join - n_post_join:,} ({(n_pre_join - n_post_join)/max(n_pre_join,1):.1%})")

    # Drop rows with missing Big Five indicators
    big_five_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    n_pre_dropnull = len(df_individual)
    df_individual = df_individual.drop_nulls(subset=big_five_traits)
    n_post_dropnull = len(df_individual)

    # --- AUDIT: after drop_nulls ---
    print(f"  [audit:after_dropnull] rows={n_post_dropnull:,}  lost={n_pre_dropnull - n_post_dropnull:,} ({(n_pre_dropnull - n_post_dropnull)/max(n_pre_dropnull,1):.1%})")

    print(f"  Individual dataset shape: {df_individual.shape}")
    print(f"  Number of participants with complete data: {len(df_individual)}")
    
    # Create zipcode-level datasets - spatial version only
    print("  Creating zipcode level datasets...")
    
    # First create a function to aggregate personality traits
    def aggregate_personality_traits(df, group_by_cols):
        # Get main traits and human-readable indicators with prefix
        main_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        
        # Find indicator columns that start with 'ind_' prefix
        indicator_columns = [col for col in df.columns 
                            if col.startswith('ind_') and
                            col not in group_by_cols]
        
        # Basic aggregation for main traits
        agg_exprs = [
            pl.col('now_zip').count().alias('participant_count'),
            *[pl.col(trait).mean().alias(f"{trait}_mean") for trait in main_traits],
            *[pl.col(trait).std().alias(f"{trait}_std") for trait in main_traits]
        ]
        
        # Add aggregation for human-readable indicators
        if indicator_columns:
            agg_exprs.extend([
                pl.col(item).mean().alias(f"{item}_mean") for item in indicator_columns
            ])
        
        return df.group_by(group_by_cols).agg(agg_exprs)
    
    # Spatial-only zipcode dataset
    print("  Creating spatial-only zipcode dataset...")
    group_by_cols = ['now_zip']
    
    # Aggregate personality traits
    df_zipcode_spatial_traits = aggregate_personality_traits(df_combined, group_by_cols)
    
    # Join with spatial segment stats
    df_zipcode_spatial = df_zipcode_spatial_traits.join(
        df_segment_stats_spatial.select([
            pl.col('zipcode'),
            *[pl.col(col) for col in df_segment_stats_spatial.columns if col != 'zipcode']
        ]),
        left_on='now_zip',
        right_on='zipcode',
        how='inner',
        coalesce=True
    )
    
    print(f"  Spatial-only zipcode dataset shape: {df_zipcode_spatial.shape}")
    print(f"  Number of unique zipcodes in spatial dataset: {len(df_zipcode_spatial)}")
    
    # Check if we need to aggregate the spatial dataset (if multiple rows per zipcode exist)
    if df_zipcode_spatial['now_zip'].n_unique() < len(df_zipcode_spatial):
        print("  Creating fully aggregated version of spatial dataset...")
        
        # Get all numeric columns for aggregation
        numeric_cols = [col for col in df_zipcode_spatial.columns
                       if col not in ['now_zip', 'zipcode', 'city']
                       and df_zipcode_spatial[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        # Create aggregation expressions
        agg_exprs = [
            # Take the most common city for each zipcode
            pl.col('city').mode().first().alias('city')
        ]
        
        # Add mean aggregations for all numeric columns
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}")
            for col in numeric_cols
        ])
        
        # Group by zipcode only
        df_zipcode_spatial_aggregated = df_zipcode_spatial.group_by('now_zip').agg(agg_exprs)
        
        print(f"  Fully aggregated spatial dataset shape: {df_zipcode_spatial_aggregated.shape}")
    else:
        # If already aggregated (one row per zipcode), just use as is
        df_zipcode_spatial_aggregated = df_zipcode_spatial
        print("  Spatial dataset already has one row per zipcode, no further aggregation needed")
    
    # For backward compatibility, set the default df_zipcode to the spatial version
    df_zipcode = df_zipcode_spatial
    
    # Similarly for aggregated version
    df_zipcode_aggregated = df_zipcode_spatial_aggregated

    # Load and merge MACH data
    print("  Loading and merging MACH data...")
    mach_path = WORKSPACE_DIR / "data/external/zip_MACH/zip_MACH_raw.shp"
    if mach_path.exists():
        # Read the shapefile and drop geometry column
        gdf_mach = gpd.read_file(str(mach_path))
        df_mach = pl.from_pandas(gdf_mach.drop(columns=['geometry']))
        
        # Rename GEOID10 to now_zip for joining
        df_mach = df_mach.rename({'GEOID10': 'now_zip'})
        
        # Merge MACH data with all zipcode datasets
        print("    Merging MACH data with zipcode datasets...")
        df_zipcode = df_zipcode.join(df_mach, on='now_zip', how='left')
        df_zipcode_spatial = df_zipcode_spatial.join(df_mach, on='now_zip', how='left')
        df_zipcode_aggregated = df_zipcode_aggregated.join(df_mach, on='now_zip', how='left')
        df_zipcode_spatial_aggregated = df_zipcode_spatial_aggregated.join(df_mach, on='now_zip', how='left')
        
        print(f"    MACH data merged successfully. Added {len(df_mach.columns) - 1} new columns.")
    else:
        print("    Warning: MACH data file not found at", mach_path)
    
    # Load and merge Census data
    print("  Loading and merging Census data...")
    census_checkpoint_path = CHECKPOINT_DIR / f"census_data_{censuspop_year}.parquet"
    if census_checkpoint_path.exists():
        df_census = pl.read_parquet(census_checkpoint_path)
        # Rename zipcode to now_zip for joining
        df_census = df_census.rename({'zipcode': 'now_zip'})
        # Remove 'area_km2' and 'total_population' before merging
        df_census = df_census.drop(['area_km2', 'total_population']) if 'area_km2' in df_census.columns and 'total_population' in df_census.columns else df_census.drop([col for col in ['area_km2', 'total_population'] if col in df_census.columns])
        print("    Merging Census data with zipcode datasets (excluding area_km2 and total_population)...")
        df_zipcode = df_zipcode.join(df_census, on='now_zip', how='left')
        df_zipcode_spatial = df_zipcode_spatial.join(df_census, on='now_zip', how='left')
        df_zipcode_aggregated = df_zipcode_aggregated.join(df_census, on='now_zip', how='left')
        df_zipcode_spatial_aggregated = df_zipcode_spatial_aggregated.join(df_census, on='now_zip', how='left')
        print(f"    Census data merged successfully. Added {len(df_census.columns) - 1} new columns (excluding area_km2 and total_population).")
    else:
        print(f"    Warning: Census data for year {censuspop_year} not found. Run Census data processing first.")
    

    # Fill null values in uscensus columns with 0
    uscensus_cols = [
        'uscensus_male_share',
        'uscensus_age_15_29_share',
        'uscensus_age_30_44_share',
        'uscensus_age_45_59_share',
        'uscensus_age_60_plus_share',
        'uscensus_population_density_km2'
    ]
    def fill_uscensus_nulls(df):
        for col in uscensus_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))
        return df

    # Add city dummies to all zipcode-level datasets
    def add_city_dummies(df):
        return pl.concat([
            df,
            df.select([
                pl.when(pl.col('city') == 'austin_texas').then(1).otherwise(0).alias('city_austin_texas'),
                pl.when(pl.col('city') == 'dallas_texas').then(1).otherwise(0).alias('city_dallas_texas'),
                pl.when(pl.col('city') == 'houston_texas').then(1).otherwise(0).alias('city_houston_texas'),
                pl.when(pl.col('city') == 'san_antonio_texas').then(1).otherwise(0).alias('city_san_antonio_texas'),
            ])
        ], how='horizontal')

    df_zipcode = fill_uscensus_nulls(add_city_dummies(df_zipcode))
    df_zipcode_spatial = fill_uscensus_nulls(add_city_dummies(df_zipcode_spatial))
    df_zipcode_aggregated = fill_uscensus_nulls(add_city_dummies(df_zipcode_aggregated))
    df_zipcode_spatial_aggregated = fill_uscensus_nulls(add_city_dummies(df_zipcode_spatial_aggregated))
    print("  ✅ City dummies added and uscensus nulls filled in all zipcode-level datasets!")

    # Return all datasets, with None for temporal datasets for backward compatibility
    return df_individual, df_zipcode, df_zipcode_spatial, None, df_zipcode_aggregated, df_zipcode_spatial_aggregated, None

def main(grid_size=100, censuspop_year=2020):
    # Check if final datasets already exist
    if (FINAL_INDIVIDUAL_DATA_PATH.exists() and 
        FINAL_ZIPCODE_DATA_PATH.exists() and
        FINAL_ZIPCODE_SPATIAL_DATA_PATH.exists() and
        FINAL_ZIPCODE_AGGREGATED_PATH.exists() and
        FINAL_ZIPCODE_SPATIAL_AGGREGATED_PATH.exists()):
        print("Loading existing final datasets...")
        df_individual = pl.read_parquet(FINAL_INDIVIDUAL_DATA_PATH)
        df_zipcode = pl.read_parquet(FINAL_ZIPCODE_DATA_PATH)
        df_zipcode_spatial = pl.read_parquet(FINAL_ZIPCODE_SPATIAL_DATA_PATH)
        df_zipcode_aggregated = pl.read_parquet(FINAL_ZIPCODE_AGGREGATED_PATH)
        df_zipcode_spatial_aggregated = pl.read_parquet(FINAL_ZIPCODE_SPATIAL_AGGREGATED_PATH)
        
        print(f"Individual dataset loaded: {df_individual.shape}")
        print(f"Zipcode dataset loaded: {df_zipcode.shape}")
        print(f"Spatial-only zipcode dataset loaded: {df_zipcode_spatial.shape}")
        print(f"Fully aggregated zipcode dataset loaded: {df_zipcode_aggregated.shape}")
        print(f"Fully aggregated spatial-only zipcode dataset loaded: {df_zipcode_spatial_aggregated.shape}")
        
        return (df_individual, df_zipcode, df_zipcode_spatial, None, 
                df_zipcode_aggregated, df_zipcode_spatial_aggregated, None)
    
    print("Processing data from scratch...")
    
    # Load survey data
    print("Loading survey data...")
    df_survey = load_survey_data()
    print(f"  [audit:survey_loaded] rows={len(df_survey):,}  now_zip_null={df_survey['now_zip'].null_count():,}")
    
    # Process GSV data and spatial join
    print("Loading zipcode shapes...")
    gdf_zipcodes = load_zipcode_shapes()
    
    print("Processing GSV metadata...")
    df_gsv = process_gsv_metadata()
    
    print("Processing segmentation results...")
    df_segments = process_segmentation_results()
    
    print("Processing flag detection results...")
    df_detections = process_detection_results()
    
    print("Processing classification results...")
    df_classifications = process_classification_results()
    
    # Process Census data for population demographics
    print("Processing US Census data...")
    df_census = process_census_data(gdf_zipcodes, year=censuspop_year)
    
    # Perform spatial join
    print(f"Performing spatial join with {grid_size}m grid size for coverage analysis...")
    df_gsv_with_zipcode = spatial_join_gsv_zipcodes(df_gsv, gdf_zipcodes, grid_size)
    
    # Aggregate segmentation and detection results
    print("Aggregating segmentation, detection, and classification results...")
    df_segment_stats = aggregate_by_zipcode(df_gsv_with_zipcode, df_segments, df_detections, df_classifications, grid_size=grid_size)
    
    # Create final datasets
    (df_individual, df_zipcode, df_zipcode_spatial, _, 
     df_zipcode_aggregated, df_zipcode_spatial_aggregated, _) = create_final_dataset(df_survey, df_segment_stats, grid_size=grid_size, censuspop_year=censuspop_year)
    
    # Save final datasets
    print("Saving final datasets...")
    (FINAL_INDIVIDUAL_DATA_PATH.parent).mkdir(parents=True, exist_ok=True)
    df_individual.write_parquet(FINAL_INDIVIDUAL_DATA_PATH)
    df_zipcode.write_parquet(FINAL_ZIPCODE_DATA_PATH)
    df_zipcode_spatial.write_parquet(FINAL_ZIPCODE_SPATIAL_DATA_PATH)
    df_zipcode_aggregated.write_parquet(FINAL_ZIPCODE_AGGREGATED_PATH)
    df_zipcode_spatial_aggregated.write_parquet(FINAL_ZIPCODE_SPATIAL_AGGREGATED_PATH)
    
    print(f"Saved individual dataset to {FINAL_INDIVIDUAL_DATA_PATH}")
    print(f"Saved default zipcode dataset to {FINAL_ZIPCODE_DATA_PATH}")
    print(f"Saved spatial-only zipcode dataset to {FINAL_ZIPCODE_SPATIAL_DATA_PATH}")
    print(f"Saved fully aggregated default zipcode dataset to {FINAL_ZIPCODE_AGGREGATED_PATH}")
    print(f"Saved fully aggregated spatial-only zipcode dataset to {FINAL_ZIPCODE_SPATIAL_AGGREGATED_PATH}")
    
    print("Dataset preparation completed!")
    return (df_individual, df_zipcode, df_zipcode_spatial, None,
            df_zipcode_aggregated, df_zipcode_spatial_aggregated, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process personality and Street View data.')
    parser.add_argument('--grid-size', type=int, default=100,
                        help='Grid cell size in meters for coverage calculation (default: 100)')
    parser.add_argument('--censuspop-year', type=int, default=2020,
                        help='Year for Census data retrieval (currently only 2020 supported, default: 2020)')
    args = parser.parse_args()
    main(grid_size=args.grid_size, censuspop_year=args.censuspop_year)