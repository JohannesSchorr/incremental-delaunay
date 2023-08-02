from unittest import TestCase, main

from incremental_delaunay.triangulation import (
    DelaunayTriangulationIncrementalWithBoundingBox,
)

from incremental_delaunay.elements import Triangle


class TestDelaunayTriangulationIncrementalWithBoundingBox(TestCase):
    def setUp(self):
        self.point_A = 0.0, 0.0
        self.point_B = 0.0, 1.0
        self.point_C = 1.0, 0.0
        self.point_D = 0.5, 0.5
        self.point_E = 0.0, 0.5
        self.point_F = 0.5, 0.0
        self.point_G = 2.0, 1.0
        self.point_H = 2.0, 0.0
        self.point_I = 1.0, 1.0
        self.points = [
            self.point_A,
            self.point_B,
            self.point_C,
            self.point_D,
            self.point_E,
            self.point_F,
            self.point_G,
        ]
        self.triangle_points = [
            self.point_A,
            self.point_B,
            self.point_C,
        ]
        self.rectangle_points = [
            self.point_A,
            self.point_B,
            self.point_C,
            self.point_I,
        ]

    def test_super_triangle(self):
        self.delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.points
        )
        point_a = 5.0, 0.5
        point_b = 1.0, 2.5
        point_c = -4.0, -2.0
        super_triangle = self.delaunay_triangulation._super_triangle()
        self.assertEqual(
            super_triangle,
            Triangle(point_a, point_b, point_c),
        )

    def test_super_rectangle(self):
        self.delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.points
        )
        super_rectangle = self.delaunay_triangulation._super_rectangle()
        self.assertEqual(
            super_rectangle,
            {
                Triangle(point_a=(-0.8, -0.4), point_b=(2.8, 1.4), point_c=(-0.8, 1.4)),
                Triangle(point_a=(-0.8, -0.4), point_b=(2.8, -0.4), point_c=(2.8, 1.4)),
            },
        )

    def test_triangulation_1_with_super_triangle(self):
        delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.triangle_points, bounding_structure="triangle"
        )

        self.assertEqual(
            delaunay_triangulation.triangles,
            {Triangle(self.point_C, self.point_A, self.point_B)},
        )

    def test_triangulation_1_with_super_rectangle(self):
        delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.triangle_points, bounding_structure="rectangle"
        )
        self.assertEqual(
            delaunay_triangulation.triangles,
            {Triangle(self.point_C, self.point_A, self.point_B)},
        )

    def test_triangulation_2_with_super_triangle(self):
        delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.rectangle_points, bounding_structure="triangle"
        )
        triangle_mesh = {
            Triangle(self.point_B, self.point_C, self.point_I),
            Triangle(self.point_C, self.point_B, self.point_A),
        }
        self.assertCountEqual(
            delaunay_triangulation.triangles,
            triangle_mesh,
        )
        self.assertCountEqual(delaunay_triangulation.points, self.rectangle_points)

    def test_triangulation_2_with_super_rectangle(self):
        delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.rectangle_points, bounding_structure="rectangle"
        )
        triangle_mesh = {
            Triangle(self.point_B, self.point_C, self.point_I),
            Triangle(self.point_C, self.point_B, self.point_A),
        }
        self.assertCountEqual(
            delaunay_triangulation.triangles,
            triangle_mesh,
        )
        self.assertCountEqual(delaunay_triangulation.points, self.rectangle_points)

    def test_triangulation_3_with_super_triangle(self):
        delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.points, bounding_structure="triangle"
        )
        self.assertCountEqual(delaunay_triangulation._triangle_points(), self.points)
        self.assertEqual(len(delaunay_triangulation.triangles), 6)

    def test_triangulation_3_with_super_rectangle(self):
        delaunay_triangulation = DelaunayTriangulationIncrementalWithBoundingBox(
            self.points, bounding_structure="rectangle"
        )
        self.assertCountEqual(delaunay_triangulation._triangle_points(), self.points)
        self.assertEqual(len(delaunay_triangulation.triangles), 6)


class DelaunayTriangulationIncrementalWithBoundingBoxTrianglulationStrainDifferenceLongitudinalShearForce(
    TestCase
):
    def setUp(self) -> None:
        self.maxDiff = None
        self.points = [
            (0.07092164560897417, 0.0),
            (0.017112718875810836, 0.0),
            (0.04368081778528715, 0.0),
            (0.0019914537568356126, 1599523.0516000006),
            (0.0557740946375556, 0.0),
            (-0.0, -0.0),
            (0.0007794542880783402, 626052.8855962341),
            (0.0006552638255870223, 0.0),
            (0.006952534477383593, 0.0),
            (0.0, 2186719.193349065),
            (0.08556008688854429, 0.0),
            (0.0, 2161363.834659677),
            (0.0, 1029329.0652537956),
            (0.0, 2196771.511936398),
            (0.0, 1568852.9374282872),
        ]

    def test_triangles(self):
        delaunay = DelaunayTriangulationIncrementalWithBoundingBox(
            self.points, bounding_structure="rectangle"
        )
        self.assertCountEqual(delaunay._triangle_points(), self.points)
        self.assertEqual(len(delaunay.triangles), 13)


if __name__ == "__main__":
    main()
