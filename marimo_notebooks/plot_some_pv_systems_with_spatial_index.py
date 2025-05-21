import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import altair as alt
    from clean_uk_pv import geospatial
    return alt, geospatial, pl


@app.cell
def _(geospatial, pl):
    pv_metadata = pl.read_csv("~/data/uk_pv/metadata.csv", try_parse_dates=True)

    gb_spatial_index = geospatial.SpatialIndex(geospatial.GBBoundingBox())

    computed_spatial_index = (
        pv_metadata.select(["latitude_rounded", "longitude_rounded"])
        .map_rows(lambda row: gb_spatial_index.lat_lon_to_index(lat=row[0], lon=row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
    )

    pv_metadata = pv_metadata.with_columns(spatial_index=computed_spatial_index)

    pv_metadata
    return gb_spatial_index, pv_metadata


@app.cell
def _(pv_metadata):
    pv_metadata.group_by("spatial_index").len().sort(by="len", descending=True)
    return


@app.cell
def _(alt, gb_spatial_index, pl, pv_metadata):
    """Plot a single grid box with its PV systems."""

    from vega_datasets import data

    SPATIAL_INDEX = 1006
    pv_systems_in_spatial_index = pv_metadata.filter(pl.col.spatial_index == SPATIAL_INDEX)

    # Create the base map for Great Britain
    # We use alt.topo_feature to extract the 'countries' features from the TopoJSON
    base_map_gb = (
        alt.Chart(alt.topo_feature(data.world_110m.url, "countries"))
        .mark_geoshape(
            fill="lightgray",
            stroke="white",
        )
        .encode(
            tooltip="id:Q",
        )
        .transform_filter(
            # The UK has id=826.
            alt.FieldEqualPredicate(field="id", equal=826)
        )
        .properties(
            width=1000,
            height=1000,
        )
    )

    _pv_systems_scatter = (
        alt.Chart(pv_systems_in_spatial_index)
        .mark_circle(
            color="black",
            opacity=0.3,
        )
        .encode(
            latitude="latitude_rounded:Q",
            longitude="longitude_rounded:Q",
            # size="kWp",
            tooltip=["ss_id", "kWp", "spatial_index", "orientation", "tilt"],
        )
    )

    bounding_box = gb_spatial_index.index_to_bounding_box(SPATIAL_INDEX)

    (
        # base_map_gb +
        bounding_box.plot() + _pv_systems_scatter
    )
    return (base_map_gb,)


@app.cell
def _(alt, base_map_gb, gb_spatial_index, geospatial, pv_metadata):
    """Plot all grid boxes that have PV systems in, and plot 10,000 PV systems."""

    spatial_indicies_for_pv_systems = (
        pv_metadata.select("spatial_index").to_series().unique().sort().to_list()
    )

    _pv_systems_scatter = (
        # Altair can't handle more than 20,000 rows of data.
        alt.Chart(pv_metadata.head(10_000))
        .mark_circle(
            color="black",
            opacity=0.3,
        )
        .encode(
            latitude="latitude_rounded:Q",
            longitude="longitude_rounded:Q",
            # size="kWp",
            tooltip=["ss_id", "kWp", "spatial_index", "orientation", "tilt"],
        )
    )

    geo_json = gb_spatial_index.geo_json(spatial_indicies_for_pv_systems)
    alt_data = geospatial.geo_json_to_altair_data(geo_json)
    (
        base_map_gb
        + _pv_systems_scatter
        + alt.Chart(alt_data).mark_geoshape(fill="lightgray", stroke="white", opacity=0.8)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
