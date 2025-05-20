import marimo

__generated_with = "0.13.10"
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
def _(pl, pv_metadata):
    SPATIAL_INDEX = 1006
    pv_systems_in_spatial_index = pv_metadata.filter(pl.col.spatial_index == SPATIAL_INDEX)
    pv_systems_in_spatial_index
    return SPATIAL_INDEX, pv_systems_in_spatial_index


@app.cell
def _(SPATIAL_INDEX, alt, gb_spatial_index, pv_systems_in_spatial_index):
    from vega_datasets import data


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
            # Using alt.FieldEqualPredicate for a robust filter
            # This specifically targets the 'name' field within the 'properties' object
            alt.FieldEqualPredicate(field="id", equal=826)
            # Or if you need to filter by multiple countries, you could use:
            # alt.FieldOneOfPredicate(field='properties.name', oneOf=['United Kingdom', 'Ireland'])
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
    return


if __name__ == "__main__":
    app.run()
