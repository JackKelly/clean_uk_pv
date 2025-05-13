import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    from datetime import datetime
    import matplotlib.pyplot as plt
    import polars as pl
    import polars_h3 as plh3
    import numpy as np
    return datetime, mo, pl, plh3, xr


@app.cell
def _(mo):
    mo.md(
        r"""
    We use a simple discrete spatial index that works like this:

    - Give a uint16 ID to each NWP pixel (just do rows first, then columns).
    - Create a GeoJSON file which provides the geospatial bounding box of each ERA5 pixel, and maps between the integer ID and the bounding box. The lat lon coords in ARCO-ERA5 refer to the centre of each grid box. Given that ERA5 is 0.25 x 0.25 degrees (from Gemini):

        Consider an ERA5 grid point with a reported latitude of 50.0° N and a longitude of 0.5° E. The bounding box for this cell would be:

        Minimum Latitude: 50.0 - 0.125 = 49.875° N
        Maximum Latitude: 50.0 + 0.125 = 50.125° N
        Minimum Longitude: 0.5 - 0.125 = 0.375° E
        Maximum Longitude: 0.5 + 0.125 = 0.625° E

    - Use [Polars-ST](https://oreilles.github.io/polars-st/api-reference/) to map from the lat lon of each PV system to the spatial index.
    - Plot with a [Choropleth map in Vega-Altair](https://altair-viz.github.io/gallery/choropleth.html).
        - For example, I could plot a count of the PV systems within each ERA5 grid cell.
    - Finding neighbours is trivial, because we've used a simple row-major numbering system. So we just do some simple integer maths.


    ## Paths not taken
    I originally wanted to use H3 as my spatial index but it appears that no H3 resolution is a particularly good fit for ERA5. To use H3, I'd either have to throw information away by using a too-low H3 resolution; or I'd have to [interpolate](https://pysal.org/tobler/notebooks/census_to_hexgrid.html) ERA5 to create a dense higher resolution H3 grid. Neither solution is very attractive.
    """
    )
    return


@app.cell
def _():
    """See here for info on H3 resolutions: https://h3geo.org/docs/core-library/restable/"""

    H3_RESOLUTION = 6
    return (H3_RESOLUTION,)


@app.cell
def _(H3_RESOLUTION, pl, plh3):
    """Open uk_pv metadata, so we can select NWP points that correspond to PV systems."""

    pv_locations = (
        pl.read_csv("~/data/uk_pv/metadata.csv")
        .with_columns(
            h3=plh3.latlng_to_cell(
                "latitude_rounded",
                "longitude_rounded",
                resolution=H3_RESOLUTION,
                return_dtype=pl.UInt64,
            )
        )["h3"]
        .unique()
        .sort()
        .implode()
    )

    pv_locations[0]
    return (pv_locations,)


@app.cell
def _():
    VARIABLES = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",  # Not sure if we need this for solar???
        "2m_temperature",
        "clear_sky_direct_solar_radiation_at_surface",
        "fraction_of_cloud_cover",
        "high_cloud_cover",
        "low_cloud_cover",
        "mean_surface_direct_short_wave_radiation_flux",
        "mean_total_precipitation_rate",
        "medium_cloud_cover",
        "snow_depth",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "total_cloud_cover",
        "total_sky_direct_solar_radiation_at_surface",
    ]
    return


@app.cell
def _(xr):
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options=dict(token="anon"),
    )
    return (ds,)


@app.cell
def _(ds):
    GB_LATITUDE = slice(59.5, 49.5)
    GB_LONGITUDE_EAST = 2
    GB_LONGITUDE_WEST = -8
    # ARCO-ERA5 represents longitude as degrees east of Greenwich.
    # So the westerly point of the UK is actually 360 degrees - 12 degrees.
    # So the ugly line below appends the longitudes in the range [360 - 12, 360) with [0, 4].
    GB_LONGITUDE = list(
        filter(lambda longitude: longitude > 360 + GB_LONGITUDE_WEST, ds.longitude.values)
    ) + list(filter(lambda longitude: longitude < GB_LONGITUDE_EAST, ds.longitude.values))
    return GB_LATITUDE, GB_LONGITUDE


@app.cell
def _(GB_LATITUDE, GB_LONGITUDE, datetime, ds):
    xr_ds = (
        ds["2m_temperature"]
        .sel(
            latitude=GB_LATITUDE,
            longitude=GB_LONGITUDE,
            time=slice(datetime(2025, 3, 1, 0, 0), datetime(2025, 3, 2, 0, 0)),
        )
        .load()
    )

    xr_ds
    return (xr_ds,)


@app.cell
def _(pl, xr_ds):
    pl.from_pandas(xr_ds.to_dataframe().reset_index()).unique(["latitude", "longitude"])
    return


@app.cell
def _(pl, plh3, xr_ds):
    df = (
        pl.from_pandas(xr_ds.to_dataframe().reset_index())
        .with_columns(
            h3=plh3.latlng_to_cell("latitude", "longitude", resolution=6, return_dtype=pl.UInt64)
        )
        .drop(["longitude", "latitude"])
        .sort(by=["time", "h3"])
    )

    df.unique("h3")
    return (df,)


@app.cell
def _(df, plh3):
    df.with_columns(h3_res_5=plh3.cell_to_parent("h3", 5))
    return


@app.cell
def _(datetime, df, pl, plh3):
    plh3.graphing.plot_hex_fills(
        df.with_columns(h3_res_4=plh3.cell_to_parent("h3", 4)).filter(
            pl.col.time == datetime(2025, 3, 1, 0, 0)
        ),
        "h3_res_4",
        "2m_temperature",
    )
    return


@app.cell
def _(df, pl, pv_locations):
    df_filtered = df.filter(pl.col.h3.is_in(pv_locations))

    df_filtered
    return (df_filtered,)


@app.cell
def _(df_filtered):
    df_filtered["h3"].unique()
    return


@app.cell
def _(datetime, df_filtered, pl, plh3):
    plh3.graphing.plot_hex_fills(
        df_filtered.filter(pl.col.time == datetime(2025, 3, 1, 0, 0)), "h3", "2m_temperature"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Next steps:

    - Load a day at a time.
    - Loop round days and NWP variables.
    - Use multiple threads to load the xarray data.
    - Store a monthly Parquet when we've loaded a month.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
