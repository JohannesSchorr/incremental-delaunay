
# Version 0.1.0

Initial version of ``incremental_delaunay``

## Features

Two implementations of an incremental delaunay triangulation
#. ``incremental_delaunay.DelaunayTriangulationIncrementalWithBoundingBox`` with a bounding box as starting point, that is deleted at the end
#. ``incremental_delaunay.DelaunayTriangulationIncremental`` with half-planes as borders of the mesh

Both implementations accept a list of points as argument.
A point is assumed to be a tuple of two floats (``tuple[float, float]``)

The triangulation is conducted during initialization.
The computed triangles are given by the attribute ``triangles`` of the above given triangulation.