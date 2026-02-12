"""Create Folium map showing PC1 and PC2 by pickup_place.

Expects a DataFrame-like object with columns: 'pickup_place', 'pc1', 'pc2'
And a zones DataFrame with columns: 'pickup_place', 'latitude', 'longitude' (or 'lat','lon').
"""
from __future__ import annotations

import os
from typing import Union

import pandas as pd
import folium


def create_pc1_pc2_map(scores: Union[pd.DataFrame, dict], zones: pd.DataFrame, output_html: str) -> str:
    """Create and save interactive Folium map.

    Parameters
    ----------
    scores: DataFrame-like
        Must contain 'pickup_place', 'pc1', 'pc2'
    zones: DataFrame
        Must contain 'pickup_place' and 'latitude' and 'longitude' (or 'lat','lon')
    output_html: str
        Path where HTML will be saved

    Returns
    -------
    str
        Path to saved HTML
    """
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(scores)

    df = scores.merge(zones, on="pickup_place", how="left")

    # Try to find lat/lon columns
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_col, lon_col = "latitude", "longitude"
    elif "lat" in df.columns and "lon" in df.columns:
        lat_col, lon_col = "lat", "lon"
    else:
        raise ValueError("zones DataFrame must contain latitude/longitude columns")

    # Map center: NYC
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

    # Normalize pc1 to color via simple linear scaling
    pc1_min, pc1_max = df['pc1'].min(), df['pc1'].max()
    pc2_abs = df['pc2'].abs()
    pc2_min, pc2_max = pc2_abs.min(), pc2_abs.max()

    for _, row in df.iterrows():
        if pd.isnull(row[lat_col]) or pd.isnull(row[lon_col]):
            continue
        # color: blue->red by pc1
        if pc1_max > pc1_min:
            frac = (row['pc1'] - pc1_min) / (pc1_max - pc1_min)
        else:
            frac = 0.5
        # convert frac to hex color (red to blue)
        r = int(255 * frac)
        b = 255 - r
        color = f"#{r:02x}00{b:02x}"

        # size by pc2 magnitude
        if pc2_max > pc2_min:
            size = 4 + 10 * (abs(row['pc2']) - pc2_min) / (pc2_max - pc2_min)
        else:
            size = 6

        folium.CircleMarker(
            location=(row[lat_col], row[lon_col]),
            radius=float(size),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{row['pickup_place']}: PC1={row['pc1']:.3f}, PC2={row['pc2']:.3f}",
        ).add_to(m)

    os.makedirs(os.path.dirname(output_html) or '.', exist_ok=True)
    m.save(output_html)
    return output_html


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--zones-csv", required=True)
    parser.add_argument("--output-html", default="./hw2_output/pc1_pc2_folium_map.html")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    zones = pd.read_csv(args.zones_csv)
    create_pc1_pc2_map(scores, zones, args.output_html)
    print("Saved map to", args.output_html)