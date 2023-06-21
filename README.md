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

## Usage 

Both of the above mentioned [Implementations](#implementations) work similarly. 
Therefore, the following points work exemplary.

```python
point_1 = 0.0, 0.0
point_2 = 1.0, 0.0
point_3 = 0.0, 1.0
points = [point_1, point_2, point_3]
```

### Incremental Delaunay with bounding box

```python
from incremental_delaunay import DelaunayTriangulationIncrementalWithBoundingBox

delaunay = DelaunayTriangulationIncrementalWithBoundingBox(points)
```

### Incremental Delaunay

```python
from incremental_delaunay import DelaunayTriangulationIncremental

delaunay = DelaunayTriangulationIncremental(points)
```

### Using the triangles

For both implementations the instances have the attribute `delaunay.triangles`. 
That gives you a set of triangles.
To look for a point within the triangles you may iterate over the triangles as follows.

```python
point = 0.3, 0.3
for triangle in delaunay.triangles: 
    if triangle.is_inside(point): 
        print(f'{triangle=}')
        break
```

## Development history

Within [m-n-kappa](https://johannesschorr.github.io/M-N-Kappa/) I looked for a way to interpolate values in a set of points.
As [bilinear Interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation) did not work the way as expected, I decided to build a mesh of triangles from the given set of points. 
A triangulation considering the Delaunay-conditions produces triangles that are optimal conditioned for interpolation. 

The Delaunay-implementation with a bounding box did also not work as expected for the given problem as some points were deleted due to their connection to the bounding box. 
The bounding box is removed at the end of the triangulation. 

Therefore, I implemented the Delaunay triangulation without a bounding box, but with half-planes at the borders of the mesh. 
Some publications call these half-planes 'ghost triangles' as the implemented algorithms work similarly.  

In ``m_n_kappa`` you may find the Delaunay-implementation not starting with a bounding structure.