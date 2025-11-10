import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -------------------------------
# Paths (update these)
# -------------------------------
your_path = Path("YOUR_SHAPEFILE_DIR")
z_path = Path("YOUR_ZONING_FILE.shp")
zc_path = Path("YOUR_CAM_ZONING_FILE.shp")

# -------------------------------
# Constants
# -------------------------------
ZONE_ORDER = ['Business', 'Industrial', 'Resource Conservation', 'Residential', 
              'Village', 'MUNI', 'Agriculture']

ZONE_COLOR = {
    'Residential': 'gold',
    'Resource Conservation': 'steelblue',
    'Agriculture': 'seagreen',
    'Business': 'maroon',
    'Industrial': 'k',
    'Village': 'salmon',
    'MUNI': 'darkgrey'
}

CLASS_MAP = {
    'Business': ['B-1', 'B-2'],
    'Industrial': ['I-1', 'I-2'],
    'Resource Conservation': ['RC'],
    'Residential': ['RR', 'RR-C', 'RR-RCA', 'SR', 'SR-RCA'],
    'Village': ['V'],
    'MUNI': ['MUNI'],
    'Agriculture': ['AC', 'AC-RCA']
}

# -------------------------------
# Helper Functions
# -------------------------------

def summarize_shapefiles(prefix: str, out_file: Path):
    """
    Summarize shapefiles starting with a prefix and save to CSV.
    """
    data = []
    for file in your_path.glob(f"{prefix}*.shp"):
        gdf = gpd.read_file(file)
        total_value = gdf['TOT_VALUE'].sum()
        num_records = len(gdf)
        total_area = gdf['Shape_Area'].sum()
        data.append([file.name, total_value, num_records, total_area])

    df = pd.DataFrame(data, columns=['Filename', 'Total Value', 'Number of Records', 'Total Area'])
    df.to_csv(out_file, index=False)
    print(df)
    return df

def compute_zone_summary(files_prefix: str, zoning: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute zoning summary for all shapefiles with a given prefix.
    """
    all_ft = pd.DataFrame()
    for file in your_path.glob(f"{files_prefix}*.shp"):
        gdf = gpd.read_file(file)
        if gdf.crs != zoning.crs:
            gdf = gdf.to_crs(zoning.crs)

        gdf['centroid'] = gdf.geometry.centroid
        centroids = gdf.set_geometry('centroid')
        joined = gpd.sjoin(centroids, zoning, how='inner', predicate='within').drop(columns='centroid')
        joined = joined.set_geometry('geometry')

        # Aggregate by CLASS_1
        agg = joined.groupby('CLASS_1').agg({'Shape_Area':'sum','STATE':'count','TOT_VALUE':'sum'}).reset_index()
        agg.rename(columns={'STATE':'count'}, inplace=True)
        agg['count_per_area'] = agg['count'] / agg['Shape_Area']

        # Ensure all CLASS_1 values exist
        all_classes = pd.DataFrame({'CLASS_1': sum(CLASS_MAP.values(), [])})
        agg_all = all_classes.merge(agg, on='CLASS_1', how='left').fillna(0)

        # Combine to main zones
        summary = pd.DataFrame({
            'Zone': list(CLASS_MAP.keys()),
            'Count': [agg_all.loc[agg_all['CLASS_1'].isin(CLASS_MAP[z]), 'count'].sum() for z in CLASS_MAP],
            'Area': [agg_all.loc[agg_all['CLASS_1'].isin(CLASS_MAP[z]), 'Shape_Area'].sum() for z in CLASS_MAP],
            'TOT_VALUE': [agg_all.loc[agg_all['CLASS_1'].isin(CLASS_MAP[z]), 'TOT_VALUE'].sum() for z in CLASS_MAP]
        })
        summary['count_per_area'] = summary['Count'] / summary['Area']
        ft_val = file.stem[3:] if file.stem[4] == '0' else file.stem[3]
        summary['ft'] = int(ft_val)
        all_ft = pd.concat([all_ft, summary])

    return all_ft

def compute_ratio_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference ratios for Count and TOT_VALUE.
    """
    df2 = pd.DataFrame()
    for zone in ZONE_ORDER:
        t = df[df['Zone'] == zone].sort_values('ft').copy()
        t['diff_count'] = t['Count'].pct_change().fillna(0)
        t['diff_value'] = t['TOT_VALUE'].pct_change().fillna(0)
        df2 = pd.concat([df2, t])
    return df2

def plot_zone_data(df: pd.DataFrame, y_cols: list, title_suffix: str):
    """
    Plot data for all zones.
    """
    for y_col in y_cols:
        plt.figure(figsize=(15, 6))
        for zone in ZONE_ORDER:
            data = df[df['Zone'] == zone].sort_values('ft')
            plt.plot(data['ft'], data[y_col], label=zone, color=ZONE_COLOR[zone], linewidth=3)
        plt.title(f"{y_col} vs ft - {title_suffix}")
        plt.xlabel("Sea Level Rise Projection (ft)")
        plt.ylabel(y_col)
        plt.grid(True)
        plt.legend(title="Zone", loc='upper left', bbox_to_anchor=(1, 1))
        plt.gca().set_facecolor('whitesmoke')
        plt.show()


# -------------------------------
# Main
# -------------------------------

# Summarize shapefiles
df_iso_summary = summarize_shapefiles('iso', your_path / "isolated_info.csv")
df_inu_summary = summarize_shapefiles('inu', your_path / "inundated_info.csv")

# Load zoning files
zoning = gpd.read_file(z_path)
zoning_cam = gpd.read_file(zc_path)

# Compute zoning summaries
all_ft_iso = compute_zone_summary('iso', zoning)
all_ft_inu = compute_zone_summary('inu', zoning)

# Compute ratio changes
all_ft_iso2 = compute_ratio_change(all_ft_iso)
all_ft_inu2 = compute_ratio_change(all_ft_inu)

# Plot results
plot_zone_data(all_ft_iso2, ['diff_count', 'diff_value'], "isolated")
plot_zone_data(all_ft_inu2, ['diff_count', 'diff_value'], "inundated")
plot_zone_data(all_ft_iso, ['Count', 'TOT_VALUE'], "isolated")
plot_zone_data(all_ft_inu, ['Count', 'TOT_VALUE'], "inundated")
