from unittest import TestCase, main

from incremental_delaunay.elements import Straight, Triangle, Halfplane
from incremental_delaunay.matrices import Matrix


class TestStraight(TestCase):
    def setUp(self) -> None:
        self.horizontal_line = Straight((1.0, 2.0), (2.0, 2.0))
        self.vertical_line = Straight((1.0, 2.0), (1.0, 1.0))
        self.diagonal_line = Straight((1.0, 1.0), (2.0, 2.0))

    def test_position_on_horizontal_line(self):
        self.assertEqual(self.horizontal_line.is_on_line((3.0, 2.0)), True)

    def test_position_on_vertical_line(self):
        self.assertEqual(self.vertical_line.is_on_line((1.0, 3.0)), True)

    def test_position_on_diagonal_line(self):
        self.assertEqual(self.diagonal_line.is_on_line((0.5, 0.5)), True)

    def test_position_on_horizontal_line_between_points(self):
        self.assertEqual(self.horizontal_line.is_between_points((1.5, 2.0)), True)

    def test_position_on_vertical_line_between_points(self):
        self.assertEqual(self.vertical_line.is_between_points((1.0, 1.5)), True)

    def test_position_on_diagonal_line_between_points(self):
        self.assertEqual(self.diagonal_line.is_between_points((1.5, 1.5)), True)

    def test_compute_y_on_horizontal_line(self):
        self.assertEqual(self.horizontal_line.compute_y(1.0), 2.0)

    def test_compute_y_on_diagonal_line(self):
        self.assertEqual(self.diagonal_line.compute_y(1.5), 1.5)

    def test_difference_to_point(self):
        line = Straight(
            (0.04368081778528715, 0.0), (0.0019914537568356126, 1599523.0516000006)
        )
        m = (1599523.0516000006) / (0.0019914537568356126 - 0.04368081778528715)
        c = 0.0 - m * 0.04368081778528715
        result = m * 0.02208007251511262 + c
        self.assertEqual(
            line.difference_to((0.02208007251511262, 799761.5258000003)),
            799761.5258000003 - result,
        )


class TestTriangle(TestCase):
    def setUp(self):
        self.point_A = 0.0, 0.0
        self.point_B = 0.0, 1.0
        self.point_C = 1.0, 0.0
        self.point_D = 1.0, 1.0
        self.triangle_1 = Triangle(self.point_A, self.point_B, self.point_C)
        self.triangle_2 = Triangle(self.point_B, self.point_A, self.point_C)
        self.triangle_3 = Triangle(self.point_C, self.point_B, self.point_A)
        self.triangle_4 = Triangle(self.point_A, self.point_C, self.point_B)
        self.triangle_5 = Triangle(self.point_B, self.point_C, self.point_D)
        self.point_E = -1.0, -1.0
        self.point_F = -1.0, 3.0000000000000004
        self.point_G = 3.0000000000000004, -1.0
        self.triangle_6 = Triangle(self.point_E, self.point_F, self.point_G)

    def test_point_inside_1(self):
        test_point = 0.5, 0.5
        self.assertEqual(self.triangle_1.is_inside(test_point), True)

    def test_point_on_edge(self):
        test_point = 0.5, 0.5
        self.assertEqual(self.triangle_1.is_on_edge(test_point), True)

    def test_point_at_edge(self):
        test_point = 0.5, 0.5
        self.assertEqual(
            self.triangle_1.at_edge(test_point), Straight(self.point_C, self.point_B)
        )

    def test_point_inside_11(self):
        test_point = 0.3, 0.3
        self.assertEqual(self.triangle_1.is_inside(test_point), True)

    def test_point_not_on_edge(self):
        test_point = 0.3, 0.3
        self.assertEqual(self.triangle_1.is_on_edge(test_point), False)

    def test_point_inside_12(self):
        test_point = 0.0, 0.3
        self.assertEqual(self.triangle_1.is_inside(test_point), True)

    def test_point_on_edge_12(self):
        test_point = 0.0, 0.3
        self.assertEqual(self.triangle_1.is_on_edge(test_point), True)

    def test_point_at_edge_12(self):
        test_point = 0.0, 0.3
        self.assertEqual(
            self.triangle_1.at_edge(test_point), Straight(self.point_A, self.point_B)
        )

    def test_point_inside_13(self):
        test_point = 0.3, 0.0
        self.assertEqual(self.triangle_1.is_inside(test_point), True)

    def test_point_on_edge_13(self):
        test_point = 0.3, 0.0
        self.assertEqual(self.triangle_1.is_on_edge(test_point), True)

    def test_point_at_edge_13(self):
        test_point = 0.3, 0.0
        self.assertEqual(
            self.triangle_1.at_edge(test_point), Straight((0.0, 0.0), (1.0, 0.0))
        )

    def test_point_inside_14(self):
        test_point = 1 / 3, 2 / 3
        self.assertEqual(self.triangle_1.is_inside(test_point), True)

    def test_point_on_edge_14(self):
        test_point = 1 / 3, 2 / 3
        self.assertEqual(self.triangle_1.is_on_edge(test_point), True)

    def test_point_at_edge_14(self):
        test_point = 1 / 3, 2 / 3
        self.assertEqual(
            self.triangle_1.at_edge(test_point), Straight(self.point_C, self.point_B)
        )

    def test_point_outside_1(self):
        test_point = 1.0, 1.0
        self.assertEqual(self.triangle_1.is_inside(test_point), False)

    def test_point_inside_2(self):
        test_point = 0.5, 0.5
        self.assertEqual(self.triangle_2.is_inside(test_point), True)

    def test_point_outside_2(self):
        test_point = 1.0, 1.0
        self.assertEqual(self.triangle_2.is_inside(test_point), False)

    def test_point_inside_3(self):
        test_point = 0.5, 0.5
        self.assertEqual(self.triangle_3.is_inside(test_point), True)

    def test_point_outside_3(self):
        test_point = 1.0, 1.0
        self.assertEqual(self.triangle_3.is_inside(test_point), False)

    def test_point_inside_4(self):
        test_point = 0.5, 0.5
        self.assertEqual(self.triangle_4.is_inside(test_point), True)

    def test_point_outside_4(self):
        test_point = 1.0, 1.0
        self.assertEqual(self.triangle_4.is_inside(test_point), False)

    def test_point_inside_5(self):
        test_point = 0.0, 0.0
        self.assertEqual(self.triangle_6.is_inside(test_point), True)

    def test_point_inside_6(self):
        test_point = 0.1, 0.1
        self.assertEqual(self.triangle_6.is_inside(test_point), True)

    def test_point_outside_5(self):
        test_point = -2.0, -2.0
        self.assertEqual(self.triangle_6.is_inside(test_point), False)

    def test_points(self):
        points = self.point_A, self.point_C, self.point_B
        self.assertEqual(self.triangle_1.points, points)

    def test_point_in_points(self):
        self.assertEqual(self.triangle_1.points[0], self.point_A)

    def test_closest_line(self):
        point = 0.0, 0.5
        self.assertEqual(
            self.triangle_1.closest_edge(point),
            tuple([self.point_B, self.point_A]),
        )

    def test_shares_edge_with(self):
        self.assertEqual(self.triangle_4.shares_edge_with(self.triangle_5), True)

    def test_shared_edge(self):
        shared_points = self.point_C, self.point_B
        self.assertEqual(self.triangle_4.shared_edge(self.triangle_5), shared_points)

    def test_switching_point_b_and_c(self):
        self.triangle = Triangle(self.point_A, self.point_B, self.point_C)
        self.assertEqual(self.triangle.point_a, self.point_A)
        self.assertEqual(self.triangle.point_b, self.point_C)
        self.assertEqual(self.triangle.point_c, self.point_B)

    def test_point_in_circumcircle_1(self):
        self.assertEqual(
            self.triangle_1.is_in_circum_circle(self.triangle_1.centroid), True
        )

    def test_point_in_circumcircle_2(self):
        point = 0.5, 0.5
        self.assertEqual(self.triangle_1.is_in_circum_circle(point), True)

    def test_point_in_circumcircle_3(self):
        point = 0.9, 0.9
        self.assertEqual(self.triangle_1.is_in_circum_circle(point), True)

    def test_point_in_circumcircle_4(self):
        point = 1.0, 1.0
        self.assertEqual(self.triangle_1.is_in_circum_circle(point), False)

    def test_point_outside_circumcircle_1(self):
        point = 1.1, 1.1
        self.assertEqual(self.triangle_1.is_in_circum_circle(point), False)

    def test_point_outside_circumcircle_2(self):
        point = -0.1, -0.1
        self.assertEqual(self.triangle_1.is_in_circum_circle(point), False)

    def test_point_outside_circumcircle_3(self):
        triangle = Triangle(
            point_c=(0.0, 0.0), point_a=(-1.0, -1.0), point_b=(3.0, -1.0)
        )
        point = 0.0, 1.0
        self.assertEqual(triangle.is_in_circum_circle(point), False)

    def test_centroid(self):
        point = 0.3333333333333333, 0.3333333333333333
        self.assertEqual(self.triangle_1.centroid, point)

    def test_circumcircle_centroid_1(self):
        point = 0.5, 0.5
        self.assertEqual(self.triangle_1.circum_circle_centroid, point)

    def test_circumcircle_centroid_2(self):
        point = 0.0, -1.0
        point_a = 0.0, 0.0
        point_b = -1.0, -1.0
        point_c = 1.0, -1.0
        triangle = Triangle(point_a, point_b, point_c)
        self.assertEqual(triangle.circum_circle_centroid, point)

    def test_circumcircle_centroid_one_side_vertical(self):
        point = 1.5, 0.5
        point_a = 0.0, 0.0
        point_b = 0.0, 1.0
        point_c = 1.0, -1.0
        triangle = Triangle(point_a, point_b, point_c)
        self.assertEqual(triangle.circum_circle_centroid, point)

    def test_circumcircle_centroid_one_side_horizontal(self):
        point = 0.5, 1.5
        point_a = 0.0, 0.0
        point_b = -1.0, 1.0
        point_c = 1.0, 0.0
        triangle = Triangle(point_a, point_b, point_c)
        self.assertEqual(triangle.circum_circle_centroid, point)

    def test_circumcircle_radius(self):
        self.assertEqual(self.triangle_1.circum_circle_radius, 0.7071067811865476)

    def test_barycentric_coordinates_point_A(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates((0.0, 0.0)), (1.0, 0.0, 0.0)
        )

    def test_barycentric_coordinates_point_B(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates((1.0, 0.0)), (0.0, 1.0, 0.0)
        )

    def test_barycentric_coordinates_point_C(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates((0.0, 1.0)), (0.0, 0.0, 1.0)
        )

    def test_barycentric_coordinates_center_bc(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates((0.5, 0.5)), (0.0, 0.5, 0.5)
        )

    def test_barycentric_coordinates_center_ab(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates((0.5, 0.0)), (0.5, 0.5, 0.0)
        )

    def test_barycentric_coordinates_center_ac(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates((0.0, 0.5)), (0.5, 0.0, 0.5)
        )

    def test_barycentric_coordinates_centroid(self):
        self.assertEqual(
            self.triangle_1.barycentric_coordinates(self.triangle_1.centroid),
            (0.3333333333333334, 0.3333333333333333, 0.3333333333333332),
        )

    def test_vector_ab(self):
        self.assertEqual(
            self.triangle_1.vector_ab,
            (
                self.point_C[0],
                self.point_C[1],
            ),  # not B as points are ordered counterclockwise
        )
        self.assertNotEqual(
            self.triangle_1.vector_ab,
            (self.point_B[0], self.point_B[1]),  # see comment above
        )

    def test_vector_ac(self):
        self.assertEqual(
            self.triangle_1.vector_ac,
            (
                self.point_B[0],
                self.point_B[1],
            ),  # not C as points are ordered counterclockwise
        )
        self.assertNotEqual(
            self.triangle_1.vector_ac,
            (self.point_C[0], self.point_C[1]),  # see comment above
        )

    def test_coeffcients(self):
        self.assertEqual(
            self.triangle_1.coefficients(),
            Matrix(
                [[self.point_C[0], self.point_B[0]], [self.point_C[1], self.point_B[1]]]
            ),
        )


class TestHalfplane(TestCase):
    def test_is_inside_horizontal_edge(self):
        halfplane = Halfplane((0.0, 0.0), (1.0, 0.0), (0.5, 0.5))
        self.assertEqual(halfplane.is_inside((0.2, 0.2)), True)

    def test_is_inside_vertical_edge(self):
        halfplane = Halfplane((0.0, 0.0), (0.0, 1.0), (0.5, 0.5))
        self.assertEqual(halfplane.is_inside((0.2, 0.2)), True)

    def test_is_inside_diagonal_edge(self):
        halfplane = Halfplane((0.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        self.assertEqual(halfplane.is_inside((0.2, 0.2)), True)
