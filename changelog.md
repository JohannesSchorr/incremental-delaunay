# Version 0.1.3

## Fixed

Parameters to test for are now
- the correct number of triangles
- the points of all triangle must match the input-points 

## Refactor

Makes now extensive use of methods provided by the [math](https://docs.python.org/3/library/math.html)-module
provided by the standard-library

# Version 0.1.2

## Fixed

Fixed some issues with the incremental delaunay with half-planes as borders.
Creates now 'better' half-planes and detects better concave hulls and adds convex triangles 
at this positions.

## Added

- more tests for the incremental Delaunay method with half-planes as borders
- [To do list](https://github.com/JohannesSchorr/incremental-delaunay/blob/master/todo.md)


# Version 0.1.1

Fixed some issues with the incremental delaunay with half-planes as borders.


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