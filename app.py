import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from streamlit_folium import st_folium


st.title("Capstone Project - Lisbon")
st.markdown("Explore and analyze road accident data from Lisbon.")

def load_data():
    df = pd.read_csv("Road_Accidents_Lisbon.csv")
    return df

df = load_data()

st.sidebar.header("Filter")

hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))

weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
selected_days = st.sidebar.multiselect("Select days of the week", options=weekday_order, default=weekday_order)


filtered_df = df[
    (df["hour"] >= hour_range[0]) & 
    (df["hour"] <= hour_range[1]) & 
    (df["weekday"].isin(selected_days))]

st.subheader(f"Temporal Patterns: Hourly Distribution ({hour_range[0]}:00 - {hour_range[1]}:00)")

fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=filtered_df.sort_values('hour'), x='hour', hue='hour', palette='Reds', legend=False, ax=ax)
ax.set_title(f'Accidents by Hour (Filtered)')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Accidents')
st.pyplot(fig)


st.divider()


st.subheader("Temporal Patterns: Weekly Distribution")

fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.countplot(data=filtered_df, x='weekday', order=[d for d in weekday_order if d in selected_days], hue='weekday', 
        palette='Blues',legend=False,ax=ax2)
      
ax2.set_title('Accidents by Day of Week (Filtered)')
ax2.set_xlabel('Day of Week')
ax2.set_ylabel('Number of Accidents')
st.pyplot(fig2)


st.divider()

st.markdown("Visualizing and clustering accidents can be really useful for instantly recognizing high-risky zones in a map. In this capstone project we detected blacks spots area in Lisbon, through the DBSCAN clustering method, given that a black spot is a area of 100m radius, with a minimum of 5 accidents and a cumulative severity of 40.")

def process_clusters(_df_input):
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                _df_input, 
                geometry=[Point(xy) for xy in zip(_df_input["longitude"], _df_input["latitude"])],
                crs="EPSG:4326")
            
            # Project to meters
            gdf_meters = gdf.to_crs(epsg=3763)
            coords = np.array(list(zip(gdf_meters.geometry.x, gdf_meters.geometry.y)))
            
            # Run DBSCAN
            db = DBSCAN(eps=100, min_samples=5, metric='euclidean').fit(coords)
            gdf["Cluster"] = db.labels_.astype(str)
            
            # Filter valid Hotspots
            cluster_stats = gdf[gdf["Cluster"] != "-1"].groupby("Cluster").agg(
                Cumulative_Severity=('Severity Index', 'sum')
            ).reset_index()
            
            valid_ids = cluster_stats[cluster_stats["Cumulative_Severity"] > 40]["Cluster"].tolist()
            return gdf[gdf["Cluster"].isin(valid_ids)].copy()

gdf_map = process_clusters(filtered_df)

center = [filtered_df["latitude"].mean(), filtered_df["longitude"].mean()]
m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

# Color Palette
cluster_ids = gdf_map["Cluster"].unique()
palette = sns.color_palette("hls", len(cluster_ids))
cluster_colors = {cluster: colors.rgb2hex(palette[i]) for i, cluster in enumerate(cluster_ids)}

# Add Cluster Areas
cluster_centers = gdf_map.groupby("Cluster").agg({'latitude': 'mean', 'longitude': 'mean'}).reset_index()

for _, center_row in cluster_centers.iterrows():
                folium.Circle(
                    location=[center_row["latitude"], center_row["longitude"]],
                    radius=100,
                    color=cluster_colors[center_row["Cluster"]],
                    fill=True, fill_opacity=0.1, weight=1, dash_array='5, 5'
                ).add_to(m)

# Add Individual Accidents
for _, row in gdf_map.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    color=cluster_colors[row["Cluster"]],
                    fill=True, fill_opacity=0.8,
                    popup=f"Cluster: {row['Cluster']}<br>Severity: {row['Severity Index']}"
                ).add_to(m)
# Render Map
st_folium(m, width=1000, height=600, returned_objects=[])

st.divider()

st.subheader("Severity Heatmap: Day vs Hour")
st.markdown("The following heatmap shows the cumulative severity without filters to provide a global overview of risk patterns.")

heatmap_data = df.pivot_table(
    index='weekday',
    columns='hour',
    values='Severity Index',
    aggfunc='sum'
).fillna(0) 


days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(days_order)

fig3, ax3 = plt.subplots(figsize=(16, 8))
sns.heatmap(
    heatmap_data,
    cmap='YlOrRd',
    annot=True,   
    fmt='.0f',
    linewidths=.5,
    ax=ax3
)

ax3.set_title('Severity Distribution: Day vs Hour', fontsize=16)
st.pyplot(fig3) 
