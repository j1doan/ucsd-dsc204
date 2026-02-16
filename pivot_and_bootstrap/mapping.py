"""Create Folium map showing PC1 and PC2 by pickup_place.

Expects a DataFrame-like object with columns: 'pickup_place', 'pc1', 'pc2'
And a zones DataFrame with columns: 'pickup_place', 'latitude', 'longitude' (or 'lat','lon').
"""
from __future__ import annotations

import os
from typing import Union

import pandas as pd
import folium


def _centroids_from_shapefile(zones_shp_path: str) -> pd.DataFrame:
    """Build a pickup_place -> (latitude, longitude) table from zone polygons."""
    import shapefile
    from pyproj import CRS, Transformer

    sf = shapefile.Reader(zones_shp_path)
    field_names = [f[0] for f in sf.fields[1:]]

    prj_path = os.path.splitext(zones_shp_path)[0] + ".prj"
    if not os.path.exists(prj_path):
        raise ValueError(f"Missing PRJ file for shapefile: {prj_path}")

    with open(prj_path, "r", encoding="utf-8") as f:
        src_crs = CRS.from_wkt(f.read())
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

    rows = []
    for sr in sf.iterShapeRecords():
        attrs = dict(zip(field_names, sr.record))
        if "LocationID" not in attrs:
            continue
        bbox = sr.shape.bbox  # xmin, ymin, xmax, ymax
        x = (bbox[0] + bbox[2]) / 2.0
        y = (bbox[1] + bbox[3]) / 2.0
        lon, lat = transformer.transform(x, y)
        rows.append(
            {
                "pickup_place": int(attrs["LocationID"]),
                "latitude": float(lat),
                "longitude": float(lon),
            }
        )

    if not rows:
        raise ValueError("No LocationID centroid rows generated from zones shapefile")

    # Some LocationIDs can appear multiple times in the shapefile; average centroids.
    centroids = pd.DataFrame(rows).groupby("pickup_place", as_index=False)[["latitude", "longitude"]].mean()
    return centroids


def create_pc1_pc2_map(
    scores: Union[pd.DataFrame, dict],
    zones: pd.DataFrame,
    output_html: str,
    zones_shp_path: str | None = None,
) -> str:
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

    # Support zone lookup files that use LocationID instead of pickup_place
    location_id_mode = False
    if "pickup_place" not in zones.columns and "LocationID" in zones.columns:
        zones = zones.rename(columns={"LocationID": "pickup_place"})
        location_id_mode = True

    if "pickup_place" not in zones.columns:
        raise ValueError("zones DataFrame must contain pickup_place or LocationID")

    # Normalize join key dtype to avoid int/float/string mismatch
    scores = scores.copy()
    zones = zones.copy()
    scores_key = pd.to_numeric(scores["pickup_place"], errors="coerce")
    zones_key = pd.to_numeric(zones["pickup_place"], errors="coerce")

    if location_id_mode:
        # Zone lookup uses integer zone IDs. Keep only integer-like score keys.
        integer_like = scores_key.notna() & (scores_key % 1 == 0)
        dropped_non_integer = int((~integer_like).sum())
        if dropped_non_integer:
            print(
                f"Dropping {dropped_non_integer} score rows with non-integer pickup_place "
                "for LocationID-based mapping."
            )
        scores = scores.loc[integer_like].copy()
        scores["pickup_place"] = scores_key.loc[integer_like].round().astype(int)
        zones["pickup_place"] = zones_key.round().astype("Int64")
    else:
        scores["pickup_place"] = scores_key
        zones["pickup_place"] = zones_key

    if ("latitude" not in zones.columns or "longitude" not in zones.columns) and zones_shp_path:
        centroids = _centroids_from_shapefile(zones_shp_path)
        zones = zones.merge(centroids, on="pickup_place", how="left")

    df = scores.merge(zones, on="pickup_place", how="left")

    # Try to find lat/lon columns
    if "latitude" in df.columns and "longitude" in df.columns:
        lat_col, lon_col = "latitude", "longitude"
    elif "lat" in df.columns and "lon" in df.columns:
        lat_col, lon_col = "lat", "lon"
    else:
        raise ValueError(
            "zones DataFrame must contain latitude/longitude columns "
            "(expected latitude/longitude or lat/lon)"
        )

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
    parser.add_argument("--zones-shp", default=None, help="Optional zones shapefile path for centroid coordinates")
    parser.add_argument("--output-html", default="./hw2_output/pc1_pc2_folium_map.html")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    zones = pd.read_csv(args.zones_csv)
    create_pc1_pc2_map(scores, zones, args.output_html, zones_shp_path=args.zones_shp)
    print("Saved map to", args.output_html)