# Incremental Delaunay triangulation

## Terms and definitions

A [triangulation](https://en.wikipedia.org/wiki/Triangulation_(geometry)) connects a given set of points leading to number of triangles in a way that no edges are crossing each other.

A [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) requires each triangle in a triangulation to fulfil the Delaunay condition.
To fulfill the **Delaunay condition** the point of a neighbouring triangle that is not point of the separating edge must be outside of the circum-circle of the triangle. 
The [circum-circle](https://en.wikipedia.org/wiki/Circumscribed_circle) goes through all three points of a triangle.

![Delaunay-condition](docs/images/delaunay_condition-light.svg)

If the point is inside the circum-circle a flip of the separating edge leads to a fulfillment of the Delaunay-condition.

## Implementations

``incremental_delaunay`` implements two approaches of incremental Delaunay triangulations: 
- starting with a bounding structure containing all points, that is removed afterwards.
- starting with three points of the set of points that are not lying on one line

## Development history

Within [m-n-kappa](https://johannesschorr.github.io/M-N-Kappa/) I looked for a way to interpolated between a set of points.
As a [bilinear Interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation) did not work the way as expected.