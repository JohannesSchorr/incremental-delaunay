from unittest import TestCase, main
import random

from incremental_delaunay.triangulation import (
    DelaunayTriangulationIncremental,
)

from incremental_delaunay.elements import Triangle


class TestDelaunayTriangulationIncremental(TestCase):
    def setUp(self):
        self.maxDiff = None
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

    def test_triangulation_one_triangle(self):
        delaunay_triangulation = DelaunayTriangulationIncremental(self.triangle_points)

        self.assertEqual(
            delaunay_triangulation.triangles,
            {Triangle(self.point_B, self.point_C, self.point_A)},
        )

    def test_triangulation_2_two_triangles_forming_a_rectangle(self):
        delaunay_triangulation = DelaunayTriangulationIncremental(self.rectangle_points)
        triangle_mesh = {
            Triangle(self.point_B, self.point_C, self.point_I),
            Triangle(self.point_B, self.point_A, self.point_C),
        }
        self.assertCountEqual(
            delaunay_triangulation.triangles,
            triangle_mesh,
        )
        self.assertCountEqual(delaunay_triangulation.points, self.rectangle_points)


class TestDelaunayTriangulationIncrementalStrainDifferenceLongitudinalShearForce(
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

    def test_dkdk(self):
        delaunay = DelaunayTriangulationIncremental(self.points)
        self.assertEqual(delaunay._triangle_points(), set(self.points))


class TestDelaunayTriangulationIncrementalMomentStrainDifference(TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.points = [
            (255154047.3662959, 0.006952534477383593),
            (268294989.4032318, 0.0557740946375556),
            (0.0, -0.0),
            (270350606.2062383, 0.07092164560897417),
            (479333253.2481214, 0.0),
            (577987747.8516432, 0.0),
            (391800487.92683905, 0.0019914537568356126),
            (347183449.2821381, 0.0),
            (272371565.9662828, 0.08556008688854429),
            (561217622.8217927, 0.0),
            (153350719.41212064, 0.0007794542880783402),
            (48766834.57919085, 0.0006552638255870223),
            (261788120.5343429, 0.017112718875810836),
            (571503418.0941486, 0.0),
            (266552138.76344815, 0.04368081778528715),
        ]

    def test_delaunay(self):
        delaunay = DelaunayTriangulationIncremental(self.points)
        self.assertEqual(delaunay._triangle_points(), set(self.points))


class TestDelaunayTriangulationIncrementalMomentStrainDifference2(TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.points = [
            (100239.77137767333, 0.0),
            (50760.87450000003, 0.003661147902990141),
            (46496.36246563134, 0.0),
            (77355.83108219982, 0.0),
            (24538.425914526517, 0.0),
            (50760.87450000003, 0.003661147902990141),
            (2922.8170404319135, 0.00011772948366581837),
            (89400.99283967182, 0.0),
            (2910.841280185274, 0.00011631398881873579),
            (0.0, -0.0),
            (89400.99283967182, 0.0),
        ]

    def test_delaunay(self):
        number_points = len(self.points)
        for index in range(len(self.points)):
            with self.subTest(index):
                delaunay = DelaunayTriangulationIncremental(
                    random.sample(self.points, number_points)
                )
                delaunay._print_triangles(do_split_types=True)
                self.assertEqual(delaunay._triangle_points(), set(self.points))
                self.assertEqual(len(delaunay.triangles), 9)


if __name__ == "__main__":
    main()
