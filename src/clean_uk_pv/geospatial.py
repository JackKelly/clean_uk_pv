"""We use a simple discrete spatial index.

- Each NWP pixel is given a unique UInt16 identifier, in row-major order, starting from the top
left.

Note that the lat lon coords in ARCO-ERA5 refer to the _centre_ of each grid box.

Given that ERA5 is 0.25 x 0.25 degrees (from Gemini):

Consider an ERA5 grid point with a reported latitude of 50.0° N and a longitude of 0.5° E. The
bounding box for this cell would be:

Minimum Latitude: 50.0 - 0.125 = 49.875° N
Maximum Latitude: 50.0 + 0.125 = 50.125° N
Minimum Longitude: 0.5 - 0.125 = 0.375° E
Maximum Longitude: 0.5 + 0.125 = 0.625° E

Finding neighbours is trivial, because we've used a simple row-major numbering system.
So we just do some simple integer maths.

## Design paths not taken
We originally wanted to use H3 as our spatial index but it appears that no
H3 resolution is a particularly good fit for ERA5. To use H3, we'd either have to throw information
away by using a too-low H3 resolution; or we'd have to
[interpolate](https://pysal.org/tobler/notebooks/census_to_hexgrid.html)
ERA5 to create a dense higher resolution H3 grid. Neither solution is very attractive.

"""

from collections.abc import Iterable


class BoundingBox:
    """Represents a geographical bounding box."""

    def __init__(self, north: float, south: float, east: float, west: float):
        """Initialize the bounding box.

        The bounding box is defined by the range [south, north), [west, east).

        Please provide longitudes in the range [-180, 180] degrees, where 0 represents the Prime
        Meridian (passing through Greenwich), positive longitudes are east of Greenwich, and
        negative longitudes are west of Greenwich.

        We do not yet support bounding boxes which cross the anti-Meridian (the line of longitude at
        180 degrees).

        Args:
          north: The northernmost latitude of the bounding box.
          south: The southernmost latitude of the bounding box.
          east: The easternmost longitude of the bounding box.
          west: The westernmost longitude of the bounding box.

        """
        # Lots of sanity checks:
        for name, value in (("west", west), ("east", east)):
            if not (-180 <= value <= 180):
                raise ValueError(
                    f"{name} of bounding box must be in the range [-180, 180) degrees, not {value}"
                )

        for name, value in (("north", north), ("south", south)):
            if not (-90 <= value <= 90):
                raise ValueError(
                    f"{name} of bounding box must be in the range [-90, 90] degrees, not {value}"
                )

        if west >= east:
            raise ValueError(
                f"west of bounding box must be less than east, but west = {west}, and east = {
                    east
                }. We do not yet support bounding boxes which span the anti-Meridian."
            )
        if north <= south:
            raise ValueError(
                f"north of bounding box must be greater than south, but north = {
                    north
                }, and south = {south}"
            )

        self.north = north
        self.south = south
        self.east = east
        self.west = west

        self.width_degrees = east - west
        self.height_degrees = north - south

    def north_south_east_west(self) -> tuple[float, float, float, float]:
        """Return the bounding box as a tuple of (north, south, east, west)."""
        return self.north, self.south, self.east, self.west

    def north_south_slice(self) -> slice:
        """Return a slice object for the north-south dimension."""
        return slice(self.north, self.south)

    def filter_longitudes_360(self, longitudes: Iterable[float]) -> Iterable[float]:
        """Filter a list of longitudes encoded as [0, 360) degrees east of Greenwich."""
        west_360 = self.west + 360 if self.west < 0 else self.west
        east_360 = self.east + 360 if self.east < 0 else self.east
        if not ((0 <= max(longitudes) < 360) and (0 <= min(longitudes) < 360)):
            raise ValueError("`longitudes` must be in the range [0, 360) degrees")
        if self.west < 0 and self.east > 0:
            # This is the case where the bounding box crosses the Prime Meridian.
            # To keep the longitudes in an west-to-east order, we have to select the
            # values west of the Prime Meridian separately:
            filtered_lons = list(filter(lambda longitude: longitude >= west_360, longitudes))
            filtered_lons += list(filter(lambda longitude: longitude < east_360, longitudes))
        else:
            # This is the case where the bounding box does not cross the Prime Meridian.
            # In this case, we can just filter the longitudes normally:
            filtered_lons = list(
                filter(lambda longitude: west_360 <= longitude < east_360, longitudes)
            )

        return filtered_lons


class GBBoundingBox(BoundingBox):
    """A bounding box for Great Britain."""

    def __init__(self):
        """Initialize the bounding box for Great Britain.

        Note that the coordinates in ERA5 specify the *central* point of each grid cell, yet our
        BoundingBox defines the *edges* of the bounding box. Hence when the edges are offset by
        0.125 degrees.

        """
        super().__init__(north=59.625, south=49.625, east=2.125, west=-8.125)


class SpatialIndex:
    """A discrete spatial index, in row-major order, starting from the top left (north west)."""

    def __init__(
        self,
        bounding_box: BoundingBox,
        resolution_degrees: float = 0.25,
    ):
        """Initialize the SpatialIndex.

        Args:
            bounding_box: :class:`BoundingBox` The bounding box to use.
            resolution_degrees: The resolution of the grid in degrees.
                For example, for ERA5, the resolution is 0.25 degrees.
                The resolution must exactly divide `bounding_box.width_degrees` and
                `bounding_box.height_degrees`.

        Raises:
            ValueError if `resolution_degrees` is not a factor of is not a factor of either the
            width or height of the bounding box.ther the width or height of the bounding box.

        """
        self.bounding_box = bounding_box
        self.resolution_degrees = resolution_degrees

        # Sanity checks:
        for attr in ["width_degrees", "height_degrees"]:
            if getattr(bounding_box, attr) % resolution_degrees != 0:
                raise ValueError(
                    f"{attr} of bounding box is not a multiple of resolution {resolution_degrees}"
                )

        # Calculate the number of rows and columns in the grid:
        self.n_rows = int(bounding_box.height_degrees / resolution_degrees)
        self.n_cols = int(bounding_box.width_degrees / resolution_degrees)

    def lat_lon_to_index(self, lat: float, lon: float) -> int:
        """Convert latitude and longitude to a grid index.

        Note that this function floors the lat/lon to the nearest north-westerly grid box, which is
        what we need for associating lat/lons with NWP grid boxes.

        Args:
            lat: The latitude of the point.
            lon: The longitude of the point.

        Returns:
            index: The spatial index.

        """
        north, south, east, west = self.bounding_box.north_south_east_west()
        if not (south <= lat < north):
            raise ValueError(f"Latitude {lat} is outside bounding box {self.bounding_box}")
        if not (west <= lon < east):
            raise ValueError(f"Longitude {lon} is outside bounding box {self.bounding_box}")
        # Calculate the row and column indices:
        row = int((north - lat) / self.resolution_degrees)
        col = int((lon - west) / self.resolution_degrees)

        # Calculate the grid index:
        return row * self.n_cols + col

    def __len__(self) -> int:
        """Return the number of grid cells in the spatial index."""
        return self.n_rows * self.n_cols
