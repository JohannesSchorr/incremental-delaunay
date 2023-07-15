"""
elements.py

.. versionadded:: 0.1.0

provides basic geometric entities for a delaunay-triangulation: 
- Straight: straight line defined by two points
- Triangle: triangle defined by three points that do not 
  lie on one line
- Halfplane: half-plane defined by three points, where the third point 
  indicates the direction of the half-plane

A point is assumed to be a a tuple of two floats
"""

import operator
from typing import Self
from dataclasses import dataclass
from functools import cached_property
from enum import Enum
from abc import ABC, abstractmethod

from .matrices import Matrix, Vector, LinearEquationsSystem


def are_points_co_linear(
    point_1: tuple[float, float],
    point_2: tuple[float, float],
    point_3: tuple[float, float],
) -> bool:
    """
    check if three given points lie on one line

    Parameters
    ----------
    point_1: tuple[float, float]
        coordinates of first point
    point_2: tuple[float, float]
        coordinates of second point
    point_3: tuple[float, float]
        coordinates of third point

    Returns
    -------
    bool
        ``True`` means all points lie on one line.
        ``False`` means points do not lie on one line.
    """
    line_1 = Straight(point_1, point_2)
    line_2 = Straight(point_2, point_3)
    if line_1.slope == line_2.slope and line_1.intercept == line_2.intercept:
        return True
    else:
        return False


class PointPosition(Enum):

    """
    Classifier for position of a point compared to a
    :py:class:`~incremental_delaunay.elements.Triangle`

    .. versionadded:: 0.1.0
    """

    EDGE_AB = 1
    EDGE_AC = 2
    EDGE_BC = 3
    INSIDE = 4
    OUTSIDE = 5


@dataclass
class Straight:

    """
    Straight line

    .. versionadded:: 0.1.0

    Parameters
    ----------
    point_1: tuple[float, float]
        first point of the straight line
    point_2: tuple[float, float]
        second point of the straight line
    """

    point_1: tuple[float, float]
    point_2: tuple[float, float]

    def __repr__(self):
        return f"Straight(point_1={self.point_1}, point_2={self.point_2})"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other) -> bool:
        if not isinstance(other, Straight):
            return False
        if self.point_1 == other.point_1 and self.point_2 == other.point_2:
            return True
        elif self.point_1 == other.point_2 and self.point_2 == other.point_1:
            return True
        else:
            return False

    @cached_property
    def y_difference(self) -> float:
        """
        difference of points in y-direction (vertical)

        .. versionadded:: 0.1.1
        """
        return abs(self.point_2[1] - self.point_1[1])

    @cached_property
    def x_difference(self) -> float:
        """
        difference of points in x-direction (horizontal)

        .. versionadded:: 0.1.1
        """
        return abs(self.point_2[0] - self.point_1[0])

    @cached_property
    def points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """points of the straight"""
        return self.point_1, self.point_2

    @cached_property
    def vector(self) -> tuple[float, float]:
        """vector of the line"""
        return self.point_2[0] - self.point_1[0], self.point_2[1] - self.point_1[1]

    @cached_property
    def slope(self) -> float:
        """slope of the straight line"""
        if self.point_1[1] == self.point_2[1]:
            return 0.0
        elif self.point_1[0] == self.point_2[0]:
            return float("inf")
        else:
            return (self.point_2[1] - self.point_1[1]) / (
                self.point_2[0] - self.point_1[0]
            )

    @cached_property
    def intercept(self) -> float:
        """intercept of the line"""
        if self.slope == float("inf"):
            return float("inf")
        else:
            return self.point_1[1] - self.slope * self.point_1[0]

    @cached_property
    def normal_slope(self) -> float:
        """slope of the normal to this line"""
        if self.slope == 0.0:
            return float("inf")
        elif self.slope == float("inf"):
            return 0.0
        else:
            return (-1.0) / self.slope

    @cached_property
    def middle_point(self) -> tuple[float, float]:
        """middle point of this straight"""
        return 0.5 * (self.point_1[0] + self.point_2[0]), 0.5 * (
            self.point_1[1] + self.point_2[1]
        )

    @cached_property
    def length(self) -> float:
        """distance between the points"""
        return (self.vector[0] ** 2.0 + self.vector[1] ** 2.0) ** 0.5

    def point_crossing_with(self, other: Self) -> tuple[float, float]:
        """find the point where this line is crossing with another"""
        if self.slope == other.slope:
            raise ValueError(
                f"Lines with identical slope ({self.slope}) can not cross each other,\n"
                f"{self},\n{other}"
            )
        if self.slope == float("inf"):
            x_point = self.point_1[0]
            y_point = other.compute_y(x_point)
        elif other.slope == float("inf"):
            x_point = other.point_1[0]
            y_point = self.compute_y(x_point)
        else:
            x_point = (other.intercept - self.intercept) / (self.slope - other.slope)
            y_point = self.compute_y(x_point)
        return x_point, y_point

    def is_on_line(self, point: tuple[float, float]):
        """check if point is on this straight"""
        if self.slope == float("inf"):
            if point[0] == self.point_1[0]:
                return True
            else:
                return False
        elif self.compute_y(point[0]) == point[1]:
            return True
        else:
            return False

    def is_between_points(self, point: tuple[float, float]) -> bool:
        """check if given ``point`` is between the points"""
        if not self.is_on_line(point):
            return False

        if self.slope == float("inf"):
            if (
                min(self.point_1[1], self.point_2[1])
                <= point[1]
                <= max(self.point_1[1], self.point_2[1])
            ):
                return True
            else:
                return False
        elif (
            0.0
            <= (point[0] - self.point_1[0]) / (self.point_2[0] - self.point_1[0])
            <= 1.0
        ):
            return True
        else:
            return False

    def compute_x(self, y: float) -> float:
        """compute ``x`` from ``y``"""
        x = (y - self.intercept) / self.slope
        return x

    def compute_y(self, x: float) -> float:
        """compute ``y`` from ``x``"""
        y = self.slope * x + self.intercept
        return y

    def normal_through(self, point: tuple[float, float]) -> Self:
        """find the normal to this Straight through the given ``point``"""
        if self.normal_slope == float("inf"):
            return Straight(point, (point[0], point[1] + 1.0))
        else:
            interception = point[1] - self.normal_slope * point[0]
            x_2 = point[0] + self.y_difference
            y_2 = x_2 * self.normal_slope + interception
            point_2 = x_2, y_2
            return Straight(point, point_2)

    def normal_through_middle(self) -> Self:
        """find the normal through the middle point"""
        return self.normal_through(self.middle_point)

    def parallel_through(self, point: tuple[float, float]) -> Self:
        """find a parallel line through the given ``point``"""
        if self.slope == float("inf"):
            return Straight(point, (point[0], point[1] + 1.0))
        else:
            interception = point[1] - self.slope * point[0]
            x_2 = point[0] + self.x_difference
            y_2 = x_2 * self.slope + interception
            point_2 = x_2, y_2
            return Straight(point, point_2)

    def difference_to(self, point: tuple[float, float]) -> float:
        """
        compute the vertical difference to the given ``point``

        In case the slope of the edge is infinite,
        the horizontal difference is measured.
        Not the 'distance' is computed, only the difference
        in a given direction (mainly vertical, sometimes horizontal).

        Parameters
        ----------
        point : tuple[float, float]
            point the difference is of interest

        Returns
        -------
        float
            difference between the point and the edge in vertical direction
        """
        if self.slope == float("inf"):
            return point[0] - self.point_1[0]
        else:
            return point[1] - self.compute_y(point[0])

    def not_shared_points(self, other: Self) -> tuple[tuple[float, float], ...]:
        """return points that are not shared between this and the ``other`` straight"""
        points = set(self.points)
        points = points.symmetric_difference(set(other.points))
        return tuple(points)

    def shared_point(self, other: Self) -> tuple[float, float]:
        """return the point this straight shares with the ``other``"""
        if other == self:
            raise ValueError
        for point in self.points:
            if point in other.points:
                return point

    def mid_normal_circle_intersections(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        compute the intersections of the normal through the middle-point
        and the circum-circle through the given points

        .. versionadded:: 0.1.2

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            two intersections between the normal through the middle-point
            and the circum-circle
        """
        r = 0.5 * self.length
        x_m, y_m = self.middle_point
        slope = self.normal_through_middle().slope
        if slope == float("inf"):
            point_1 = x_m, y_m + r
            point_2 = x_m, y_m - r
        else:
            intercept = self.normal_through_middle().intercept
            a = 1.0 + slope**2
            b = -2.0 * x_m + 2.0 * (intercept - y_m) * slope
            p = b / a
            c = (x_m**2.0 + (intercept - y_m) ** 2.0 - r**2.0) / a
            q = c / a
            x_1 = -p / 2.0 + ((p / 2.0) ** 2.0 - q) ** 0.5
            y_1 = self.normal_through_middle().compute_y(x_1)
            point_1 = x_1, y_1
            x_2 = -p / 2.0 - ((p / 2.0) ** 2.0 - q) ** 0.5
            y_2 = self.normal_through_middle().compute_y(x_2)
            point_2 = x_2, y_2
        return point_1, point_2


class MetaTriangle(ABC):

    """
    Meta-Class for Triangles used for Delaunay-Triangulation

    .. versionadded:: 0.1.0

    """

    def __init__(
        self,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        point_c: tuple[float, float],
    ):
        """
        Parameters
        ----------
        point_a : tuple[float, float]
            first point of the triangle
        point_b : tuple[float, float]
            second point of the triangle
        point_c : tuple[float, float]
            third point of the triangle
        """
        self._point_a = point_a
        self._point_b = point_b
        self._point_c = point_c
        self._last_point_position = None, None
        self._check_points()

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, MetaTriangle):
            return False
        for point in self.points:
            if point not in other.points:
                return False
        return True

    def _check_points(self):
        if self._point_a == self._point_b:
            raise ValueError(f"point_a={self.point_a}==point_b={self.point_b}")
        elif self._point_b == self._point_c:
            raise ValueError(f"point_b={self.point_b}==point_c={self.point_c}")
        elif self._point_c == self._point_a:
            raise ValueError(f"point_c={self.point_c}==point_a={self.point_a}")

    def __hash__(self):
        return hash(str(self))

    @property
    def point_a(self) -> tuple[float, float]:
        """first point of the triangle"""
        return self._point_a

    @property
    def point_b(self) -> tuple[float, float]:
        """second point of the triangle"""
        return self._point_b

    @property
    def point_c(self) -> tuple[float, float]:
        """third point of the triangle"""
        return self._point_c

    @cached_property
    def points(self):
        """points of the triangle"""
        return self._point_a, self._point_b, self._point_c

    @cached_property
    def edge_ab(self) -> Straight:
        """edge between point a and b"""
        return Straight(self.point_a, self.point_b)

    @cached_property
    def edge_bc(self) -> Straight:
        """edge between point b and c"""
        return Straight(self.point_b, self.point_c)

    @cached_property
    def edge_ca(self) -> Straight:
        """edge between point c and a"""
        return Straight(self.point_c, self.point_a)

    @cached_property
    def edges(self) -> tuple[Straight, Straight, Straight]:
        """edges of the triangle"""
        return (
            self.edge_ab,
            self.edge_bc,
            self.edge_ca,
        )

    def is_edge(self, edge: Straight) -> bool:
        """check if given ``edge`` is edge of this triangle"""
        if edge in self.edges:
            return True
        else:
            return False

    @abstractmethod
    def is_inside(self, point: tuple[float, float]) -> bool:
        """
        check if the given point is inside the triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being inside the triangle

        Returns
        -------
        bool
            ``True`` means the ``point`` is inside the triangle.
            ``False`` indicated the ``point`` is outside / not inside the triangle.
        """
        ...

    @abstractmethod
    def split(self, point: tuple[float, float]):
        """
        split this triangle at the given point into two or three triangles

        If the ``point`` lies within the triangle three triangles are
        created meeting at the given ``point``.
        In case the ``point`` lies on the edge of the triangle
        two triangles are created also meeting at that point.
        In the latter case please be aware that the
        neighbouring triangle at this edge must also be split.

        Parameters
        ----------
        point : tuple[float, float]
             splitting point of the triangle

        Returns
        -------
        set[Triangle]
            triangles split at the given ``point``
        """
        ...

    @abstractmethod
    def at_edge(self, point: tuple[float, float]) -> Straight:
        """
        edge where the point is located at

        Parameters
        ----------
        point : tuple[float, float]
            point that is located at an edge of the triangle

        Returns
        -------
        Straight
            edge the given ``point`` lies on
        """
        ...

    @abstractmethod
    def is_in_circum_circle(
        self, point: tuple[float, float], by: str = "geometry"
    ) -> bool:
        """check if the given point is inside the circum-circle of triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being inside the circum-circle of the triangle
        by : str
            Choose method to compute the check.
            Possible values are:

            - ``'geometry'``: compares the distance of the given point
              to the circum-circle-center with the circum-circle-center
            - ``'determinant'``: computes the determinant

        Returns
        -------
        bool
            ``True`` means the ``point`` is inside the circum-circle of the triangle.
            ``False`` indicates the ``point`` is outside / not inside
            the circum-circle of the triangle.
        """
        ...

    @cached_property
    def circum_circle_radius(self) -> float:
        """radius of the circum-circle"""
        return (
            sum(
                (
                    (position - self.circum_circle_centroid[index]) ** 2.0
                    for index, position in enumerate(self.point_a)
                )
            )
        ) ** 0.5

    @property
    @abstractmethod
    def circum_circle_centroid(self) -> tuple[float, float]:
        """compute the circum_centroid of the circum_circle of the triangle"""
        ...

    @abstractmethod
    def is_on_edge(self, point: tuple[float, float]):
        """
        check if given point lies on an edge of the triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being on an edge of the triangle

        Returns
        -------
        bool
            ``True`` means the ``point`` is on an edge of the triangle.
            ``False`` indicated the ``point`` is not on an edge of the triangle.
        """
        ...

    def shares_edge_with(self, triangle: Self) -> bool:
        """
        check if two similar points are given in both triangles

        Parameters
        ----------
        triangle : MetaTriangle
            the other triangle

        Returns
        -------
        bool
            ``True`` if two points are the same.
            In all other cases ``False`` is given.
        """
        if len(self.shared_edge(triangle)) == 2:
            return True
        else:
            return False

    def shared_edge(self, triangle: Self) -> tuple[tuple[float, float], ...]:
        """
        find the points that are shared between this and the other triangle

        Parameters
        ----------
        triangle : MetaTriangle
            other triangle that shares an edge

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            points of the shared edge
        """
        return tuple((point for point in self.points if point in triangle.points))

    def not_shared_point(self, triangle: Self | Straight) -> tuple[float, float]:
        """
        point that is not shared with the other ``triangle``

        Parameters
        ----------
        triangle : MetaTriangle | Straight
            other triangle

        Returns
        -------
        tuple[float, float]
            point that is not shared with the other ``triangle``
        """
        for point in self.points:
            if point not in triangle.points:
                return point

    def shared_points(self, triangle: Self) -> list[tuple[float, float]]:
        """
        shared points

        Parameters
        ----------
        triangle : MetaTriangle
            triangle to check for shared points

        Returns
        -------
        list[tuple[float, float]]
            shared points
        """
        return [point for point in self.points if point in triangle.points]

    def shares_point(
        self, points: list[tuple[float, float]] | set[tuple[float, float]] | Self
    ) -> bool:
        """
        check if at least one point is shared

        Parameters
        ----------
        points : list[tuple[float, float]] | set[tuple[float, float]] | Triangle
            points or a :py:class:`~incremental_delaunay.elements.Triangle`,
            where the points are extracted.

        Returns
        -------
        bool
            ``True` if both triangles share at least one point.
            ``False`` if no points are shared.
        """
        if isinstance(points, MetaTriangle):
            if not self.shared_points(points):
                return False
            else:
                return True
        elif isinstance(points, (list, set)):
            for point in points:
                if point in self.points:
                    return True
            return False
        else:
            print("Error")
            return False

    def not_shared_points(self, triangle: Self) -> tuple[tuple[float, float], ...]:
        """
        points that are not shared between two triangles

        Parameters
        ----------
        triangle : Triangle
            other triangle

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            points that are not shared
        """
        shared_points = self.shared_edge(triangle)
        return tuple(
            (
                point
                for point in self.points + triangle.points
                if point not in shared_points
            )
        )

    def is_neighbour(self, other: Self) -> bool:
        """
        determine if this instance and the other instance are direct
        neighbours

        .. versionadded:: 0.1.1

        Neighbourhood is defined by sharing two points

        Parameters
        ----------
        other : MetaTriangle
            other triangle to check for neighbourhood

        Returns
        -------
        bool
            ``True`` if this and the ``other`` triangle are neighbours
        """
        if len(self.shared_points(other)) == 2:
            return True
        else:
            return False


class Triangle(MetaTriangle):

    """
    Triangle

    .. versionadded:: 0.1.0
    """

    def __init__(
        self,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        point_c: tuple[float, float],
    ):
        """
        Parameters
        ----------
        point_a : tuple[float, float]
            first point of the triangle
        point_b : tuple[float, float]
            second point of the triangle
        point_c : tuple[float, float]
            third point of the triangle

        Examples
        --------
        Our exemplary triangle has the following three points.

        >>> point_1 = 0.0, 0.0
        >>> point_2 = 0.0, 1.0
        >>> point_3 = 1.0, 0.0

        By passing these points to :py:class:`~incremental_delaunay.elements.Triangle`
        our ``triangle`` is initialized.

        >>> from incremental_delaunay.elements import Triangle
        >>> triangle = Triangle(
        ...    point_a=point_1,
        ...    point_B=point_2,
        ...    point_C=point_3,
        ... )

        Now we check several properties.

        >>> triangle.circum_circle_centroid
        0.3333333333333333, 0.3333333333333333

        The circum-circle goes through each of the given points of the
        triangle.
        Its circum_centroid is given by
        :py:attr:`~incremental_delaunay.elements.Triangle.circum_circle_centroid`.

        >>> triangle.circum_circle_centroid
        0.5, 0.5

        The radius of the circum-circle is given by
        :py:attr:`~incremental_delaunay.elements.Triangle.circum_circle_radius`.

        >>> triangle.circum_circle_radius
        0.7071067811865476

        In the `delaunay-triangulation <https://en.wikipedia.org/wiki/Delaunay_triangulation#Algorithms>`_
        no point of a neighbouring triangle must lie within this circum-circle.
        You can check if an arbitrary point lies within the circum-circle by using
        :py:meth:`~incremental_delaunay.elements.Triangle.is_in_circum_circle`

        >>> point_inside_circle = 0.5, 0.5
        >>> triangle.is_in_circum_circle(point_inside_circle)
        True

        >>> point_outside_circle = 1.1, 1.1
        >>> triangle.is_in_circum_circle(point_outside_circle)
        False

        If you want to check if a point lies within your ``triangle``
        you may use
        :py:meth:`~incremental_delaunay.elements.Triangle.is_inside`

        >>> point_inside = 0.2, 0.2
        >>> triangle.is_inside(point_inside)
        True

        >>> point_outside = 0.6, 0.6
        >>> triangle.is_inside(point_outside)
        False
        """
        super().__init__(point_a, point_b, point_c)
        self.__post_init__()

    def __repr__(self):
        return (
            f"Triangle("
            f"point_a={self.point_a}, "
            f"point_b={self.point_b}, "
            f"point_c={self.point_c})"
        )

    def __post_init__(self):
        if not self.points_are_counterclockwise():
            self._point_b, self._point_c = self.point_c, self.point_b
        if are_points_co_linear(self.point_a, self.point_b, self.point_c):
            raise ValueError(f"points of {self} are co-linear (lie on one line)")

    def points_are_counterclockwise(self) -> bool:
        """check if points rotate counterclockwise around circum_centroid"""
        direction = (self.point_a[0] - self.centroid[0]) * (
            self.point_b[1] - self.centroid[1]
        ) - (self.point_a[1] - self.centroid[1]) * (self.point_b[0] - self.centroid[0])
        if direction < 0:
            return False
        else:
            return True

    @cached_property
    def vector_ab(self) -> tuple[float, float]:
        """
        vector between :py:attr:`~incremental_delaunay.elements.Triangle.point_a`
        and :py:attr:`~incremental_delaunay.elements.Triangle.point_B`
        """
        return self.edge_ab.vector

    @cached_property
    def center_ab(self) -> tuple[float, float]:
        """
        center point of the edge between :py:attr:`~incremental_delaunay.elements.Triangle.point_a`
        and :py:attr:`~incremental_delaunay.elements.Triangle.point_B`
        """
        return self.edge_ab.middle_point

    @cached_property
    def vector_ac(self) -> tuple[float, float]:
        """
        vector between :py:attr:`~incremental_delaunay.elements.Triangle.point_a`
        and :py:attr:`~incremental_delaunay.elements.Triangle.point_C`
        """
        return self.point_c[0] - self.point_a[0], self.point_c[1] - self.point_a[1]

    @cached_property
    def center_ac(self) -> tuple[float, float]:
        """center point of the edge between :py:attr:`~incremental_delaunay.elements.Triangle.point_a`
        and :py:attr:`~incremental_delaunay.elements.Triangle.point_C`"""
        vector_ac = self.vector_ac
        return (
            self.point_a[0] + 0.5 * vector_ac[0],
            self.point_a[1] + 0.5 * vector_ac[1],
        )

    def coefficients(self) -> Matrix:
        """matrix of the vectors :py:attr:`~incremental_delaunay.elements.Triangle.vector_ab`
        and :py:attr:`~incremental_delaunay.elements.Triangle.vector_ac`"""
        return Matrix(
            [
                [self.edge_ab.vector[0], -self.edge_ca.vector[0]],
                [self.edge_ab.vector[1], -self.edge_ca.vector[1]],
            ]
        )

    @cached_property
    def _barycentric_determinant(self) -> float:
        """determinant for computing barycentric coordinates"""
        return (self.point_b[1] - self.point_c[1]) * (
            self.point_a[0] - self.point_c[0]
        ) + (self.point_c[0] - self.point_b[0]) * (self.point_a[1] - self.point_c[1])

    def barycentric_coordinates(
        self, point: tuple[float, float]
    ) -> tuple[float, float, float]:
        """determine the barycentric coordinates of the given ``point``"""
        coord_a = (
            (self.point_b[1] - self.point_c[1]) * (point[0] - self.point_c[0])
            + (self.point_c[0] - self.point_b[0]) * (point[1] - self.point_c[1])
        ) / self._barycentric_determinant
        coord_b = (
            (self.point_c[1] - self.point_a[1]) * (point[0] - self.point_c[0])
            + (self.point_a[0] - self.point_c[0]) * (point[1] - self.point_c[1])
        ) / self._barycentric_determinant
        coord_c = 1.0 - coord_a - coord_b
        return coord_a, coord_b, coord_c

    def _compute_parameters(self, point: tuple[float, float]) -> tuple[float, float]:
        """compute parameters to check if ``point`` is inside or on the edge of the triangle"""
        constants = Vector([point[0] - self.point_a[0], point[1] - self.point_a[1]])
        parameters = LinearEquationsSystem(self.coefficients(), constants).solve("LU")
        return tuple(parameters.entries)

    def _position_of(self, point: tuple[float, float]) -> Enum | None:
        """determine the position of the given ``point`` compared to the triangle"""
        if point == self._last_point_position[0]:
            return self._last_point_position[1]
        parameters = self._compute_parameters(point)
        if parameters[0] == 0.0 and 0.0 < parameters[1] < 1.0:
            self._last_point_position = point, PointPosition.EDGE_AC
        elif 0.0 < parameters[0] < 1.0 and parameters[1] == 0.0:
            self._last_point_position = point, PointPosition.EDGE_AB
        elif (
            round(sum(parameters), 10) == 1.0
            and 0.0 <= parameters[0] <= 1.0
            and 0.0 <= parameters[1] <= 1.0
        ):
            self._last_point_position = point, PointPosition.EDGE_BC
        elif (
            0.0 < parameters[0] < 1.0
            and 0.0 < parameters[1] < 1.0
            and 0.0 < sum(parameters) < 1.0
        ):
            self._last_point_position = point, PointPosition.INSIDE
        else:
            self._last_point_position = point, PointPosition.OUTSIDE
        return self._last_point_position[1]

    def is_inside(self, point: tuple[float, float]) -> bool:
        """
        check if the given point is inside the triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being inside the triangle

        Returns
        -------
        bool
            ``True`` means the ``point`` is inside the triangle.
            ``False`` indicated the ``point`` is outside / not inside the triangle.
        """
        if self._position_of(point) in [
            PointPosition.EDGE_AB,
            PointPosition.EDGE_AC,
            PointPosition.EDGE_BC,
            PointPosition.INSIDE,
        ]:
            return True
        else:
            return False

    def split(self, point: tuple[float, float]) -> set[Self]:
        """
        split this triangle at the given point into two or three triangles

        If the ``point`` lies within the triangle three triangles are
        created meeting at the given ``point``.
        In case the ``point`` lies on the edge of the triangle
        two triangles are created also meeting at that point.
        In the latter case please be aware that the
        neighbouring triangle at this edge must also be split.

        Parameters
        ----------
        point : tuple[float, float]
             splitting point of the triangle

        Returns
        -------
        set[Triangle]
            triangles split at the given ``point``
        """
        position = self._position_of(point)
        match position:
            case PointPosition.EDGE_AB:
                return {
                    Triangle(self.point_c, point, self.point_a),
                    Triangle(self.point_c, point, self.point_b),
                }
            case PointPosition.EDGE_AC:
                return {
                    Triangle(self.point_b, point, self.point_a),
                    Triangle(self.point_b, point, self.point_c),
                }
            case PointPosition.EDGE_BC:
                return {
                    Triangle(self.point_a, point, self.point_b),
                    Triangle(self.point_a, point, self.point_c),
                }
            case PointPosition.INSIDE:
                return {
                    Triangle(self.point_a, self.point_b, point),
                    Triangle(self.point_b, self.point_c, point),
                    Triangle(self.point_c, self.point_a, point),
                }
            case _:
                return {self}

    def at_edge(self, point: tuple[float, float]) -> Straight:
        """
        edge where the point is located at

        Parameters
        ----------
        point : tuple[float, float]
            point that is located at an edge of the triangle

        Returns
        -------
        Straight
            edge the given ``point`` lies on
        """
        position = self._position_of(point)
        match position:
            case PointPosition.EDGE_AB:
                return self.edge_ab
            case PointPosition.EDGE_AC:
                return self.edge_ca
            case PointPosition.EDGE_BC:
                return self.edge_bc
            case _:
                print("point does not lie on an edge")

    def is_on_edge(self, point: tuple[float, float]) -> bool:
        """
        check if given point lies on an edge of the triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being on an edge of the triangle

        Returns
        -------
        bool
            ``True`` means the ``point`` is on an edge of the triangle.
            ``False`` indicated the ``point`` is not on an edge of the triangle.
        """
        position = self._position_of(point)
        if position in [
            PointPosition.EDGE_AB,
            PointPosition.EDGE_AC,
            PointPosition.EDGE_BC,
        ]:
            return True
        else:
            return False

    def is_in_circum_circle(
        self, point: tuple[float, float], by: str = "geometry"
    ) -> bool:
        """check if the given point is inside the circum-circle of triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being inside the circum-circle of the triangle
        by : str
            Choose method to compute the check.
            Possible values are:

            - ``'geometry'``: compares the distance of the given point
              to the circum-circle-center with the circum-circle-center
            - ``'determinant'``: computes the determinant

        Returns
        -------
        bool
            ``True`` means the ``point`` is inside the circum-circle of the triangle.
            ``False`` indicates the ``point`` is outside / not inside
            the circum-circle of the triangle.
        """
        if point in self.points:
            return False
        if by.upper() == "GEOMETRY":
            return self._is_in_circum_circle_by_geometry(point)
        elif by.upper() == "DETERMINANT":
            return self._is_in_circum_circle_by_determinant(point)

    def _is_in_circum_circle_by_determinant(self, point: tuple[float, float]) -> bool:
        """
        check if the given ``point`` is in circum-circle of the triangle
        using the determinant
        """
        matrix = Matrix(
            [
                self._circum_circle_matrix_row(self.point_a),
                self._circum_circle_matrix_row(self.point_b),
                self._circum_circle_matrix_row(self.point_c),
                self._circum_circle_matrix_row(point),
            ]
        )
        determinant = matrix.determinant()
        if determinant > 0.0:
            return True
        else:
            return False

    def _is_in_circum_circle_by_geometry(self, point: tuple[float, float]) -> bool:
        """
        check if the given ``point`` is in circum-circle of the triangle
        comparing the circum_centroid of the circum-circle and its radius
        """
        distance = (
            sum(
                (
                    (position - self.circum_circle_centroid[index]) ** 2.0
                    for index, position in enumerate(point)
                )
            )
        ) ** 0.5
        if distance < self.circum_circle_radius:
            return True
        else:
            return False

    @staticmethod
    def _circum_circle_matrix_row(point: tuple[float, float]) -> list[float]:
        """
        helper-function to method
        :py:meth:`~incremental_delaunay.elements.Triangle._is_in_circum_circle_by_determinant`
        """
        return [point[0], point[1], point[0] ** 2.0 + point[1] ** 2.0, 1.0]

    @cached_property
    def circum_circle_centroid(self) -> tuple[float, float]:
        """
        circum_centroid of the circum_circle of the triangle

        equals to the crossing-point of the normals through the centers of the edges
        """

        ab_center_normal = self.edge_ab.normal_through_middle()
        ca_center_normal = self.edge_ca.normal_through_middle()
        try:
            return ab_center_normal.point_crossing_with(ca_center_normal)
        except ValueError:
            bc_center_normal = self.edge_bc.normal_through_middle()
            return ab_center_normal.point_crossing_with(bc_center_normal)

    def closest_edge(
        self, point: tuple[float, float]
    ) -> tuple[tuple[float, float], ...]:
        """
        find the edge of the triangle closest to the given point

        Parameters
        ----------
        point : tuple[float, float]
            the point where the closest line is to be found

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            points representing the closest edge
        """
        lines = []
        for index, point_1 in enumerate(self.points):
            if index == 2:
                point_2 = self.points[0]
            else:
                point_2 = self.points[index + 1]
            distance = (
                abs(
                    (point_2[0] - point_1[0]) * (point_1[1] - point[1])
                    - (point_1[0] - point[0]) * (point_2[1] - point_1[1])
                )
                / ((point_2[0] - point_1[0]) ** 2.0 + (point_2[1] - point_1[1]) ** 2.0)
                ** 0.5
            )
            lines.append({"distance": distance, "points": tuple([point_1, point_2])})
        return min(lines, key=operator.itemgetter("distance"))["points"]

    @cached_property
    def centroid(self) -> tuple[float, float]:
        """circum_centroid of the triangle"""
        points = [self.point_a, self.point_b, self.point_c]
        x = sum((point[0] for point in points)) / 3.0
        y = sum((point[1] for point in points)) / 3.0
        return x, y

    def scale(self, factor: float = 1.1) -> Self:
        """
        scale the triangle by the given factor

        Parameters
        ----------
        factor : float
            Factor to scale the triangle (Default: 1.1).
            ``factor`` must be greater zero, otherwise an ``ValueError`` is raised
            ``factor=1.0`` has no effect.

        Returns
        -------
        Triangle
            scaled triangle

        Raises
        ------
        ValueError
            if ``factor`` is smaller or equal zero
        """
        if factor <= 0.0:
            raise ValueError('argument "factor" must be greater zero')
        new_points = []
        for point in self.points:
            x = self.centroid[0] + factor * (point[0] - self.centroid[0])
            y = self.centroid[1] + factor * (point[1] - self.centroid[1])
            new_point = x, y
            new_points.append(new_point)
        return Triangle(new_points[0], new_points[1], new_points[2])

    def outside_point(
        self,
        edge: Straight,
    ) -> tuple[float, float]:
        """
        determine a point outside the triangle on the side of the given ``edge``

        Parameters
        ----------
        edge : tuple[tuple[float, float], tuple[float, float]]
            edge of the triangle the point must be near of

        Returns
        -------
        tuple[float, float]
        """
        point_1, point_2 = edge.mid_normal_circle_intersections()
        opposite_point = self.not_shared_point(edge)
        if edge.difference_to(point_1) * edge.difference_to(opposite_point) < 0.0:
            outside_point = point_1
        elif edge.difference_to(point_2) * edge.difference_to(opposite_point) < 0.0:
            outside_point = point_2
        else:
            raise ValueError
        if edge.slope != 0.0 and edge.slope != float("inf"):
            line = Straight(edge.middle_point, outside_point)
            x_factors = [
                (point[0] - edge.middle_point[0]) / line.vector[0]
                for point in edge.points
            ]
            y_factors = [
                (point[1] - edge.middle_point[1]) / line.vector[1]
                for point in edge.points
            ]
            positive_factors = [
                factor for factor in x_factors + y_factors if factor > 0.0
            ]
            factor = min(positive_factors)
            outside_point = (
                edge.middle_point[0] + factor * line.vector[0],
                edge.middle_point[1] + factor * line.vector[1],
            )
        return outside_point


class Halfplane(MetaTriangle):

    """
    Halfplane-Triangle

    .. versionadded:: 0.1.0
    """

    def __init__(
        self,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        point_c: tuple[float, float],
    ):
        super().__init__(point_a, point_b, point_c)
        self._point_c = self._adjust_point_c()

    def __repr__(self):
        return (
            f"Halfplane("
            f"point_a={self.point_a}, "
            f"point_b={self.point_b}, "
            f"point_c={self.point_c})"
        )

    def _adjust_point_c(self) -> tuple[float, float]:
        normal_line = self.edge_ab.normal_through_middle()
        parallel_line = self.edge_ab.parallel_through(self.point_c)
        return normal_line.point_crossing_with(parallel_line)

    @cached_property
    def edge_ab(self) -> Straight:
        """compute slope and intercept of edge ab"""
        return Straight(self.point_a, self.point_b)

    @cached_property
    def normal_at_point_a(self) -> Straight:
        """determine the normal at point A"""
        return self.edge_ab.normal_through(self.point_a)

    @cached_property
    def normal_at_point_b(self) -> Straight:
        """determine the normal at point A"""
        return self.edge_ab.normal_through(self.point_b)

    @cached_property
    def point_c_difference_to_edge_ab(self) -> float:
        """difference between point_c to the edge ab"""
        return self.edge_ab.difference_to(self.point_c)

    def is_neighbour(self, halfplane: Self) -> bool:
        """
        check if this and the other halfplane are neighbours /
        share one point
        """
        if halfplane == self:
            return False
        for point in self.edge_ab.points:
            if point in halfplane.edge_ab.points:
                return True
        return False

    def _position_of(self, point: tuple[float, float]) -> Enum | None:
        """
        determine where the given ``point`` is positioned compared to
        the half-plane

        Possible values are:
        - on the edge of the half-plane
        - on the side of point 'c' compared to the edge
        - on the other side of point 'c' compared to the edge

        Parameters
        ----------
        point : tuple[float, float]
            point to check for its position compared to the half-plane

        Returns
        -------
        Enum
            position-definition
        """
        if point == self._last_point_position[0]:
            return self._last_point_position[1]
        if self.edge_ab.is_between_points(point):
            self._last_point_position = point, PointPosition.EDGE_AB
            return PointPosition.EDGE_AB

        if self.edge_ab.is_on_line(point):
            self._last_point_position = point, PointPosition.OUTSIDE
            return PointPosition.OUTSIDE

        point_difference_to_edge_ab = self.edge_ab.difference_to(point)
        sign_1 = point_difference_to_edge_ab / abs(point_difference_to_edge_ab)
        sign_2 = self.point_c_difference_to_edge_ab / abs(
            self.point_c_difference_to_edge_ab
        )
        side_factor = sign_1 * sign_2

        if side_factor > 0.0:
            self._last_point_position = point, PointPosition.INSIDE
        else:
            self._last_point_position = point, PointPosition.OUTSIDE
        return self._last_point_position[1]

    def is_inside(self, point: tuple[float, float]) -> bool:
        """
        check if the given point is inside the halfplane

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being inside the triangle

        Returns
        -------
        bool
            ``True`` means the ``point`` is inside the halfplane.
            ``False`` indicated the ``point`` is outside / not inside the halfplane.
        """
        position = self._position_of(point)
        match position:
            case PointPosition.EDGE_AB:
                return True
            case PointPosition.INSIDE:
                return True
            case PointPosition.OUTSIDE:
                return False

    def is_on_edge(self, point: tuple[float, float]):
        """
        check if given point lies on an edge of the triangle

        Parameters
        ----------
        point : tuple[float, float]
            this point is tested for being on an edge of the triangle

        Returns
        -------
        bool
            ``True`` means the ``point`` is on an edge of the triangle.
            ``False`` indicated the ``point`` is not on an edge of the triangle.
        """
        position = self._position_of(point)
        if position == PointPosition.EDGE_AB:
            return True
        else:
            return False

    def split(self, point: tuple[float, float]) -> set[Triangle | Self]:
        """
        split this half-plane at the given point into one triangle and
        at least one halfplane

        If the ``point`` lies within the halfplane one triangle is
        created with the edge of the halfplane and the given ``point``.
        If the ``point`` lies within the normals of the edge-points two
        half-planes are created.
        In case the ``point`` lies outside the normals only one Halfplane
        is created.

        In case the ``point`` lies on the edge of the halfplane
        two half-planes are created also meeting at that point.
        In the latter case please be aware that the
        neighbouring triangle at this edge must also be split.

        Parameters
        ----------
        point : tuple[float, float]
             splitting point of the halfplane

        Returns
        -------
        set[Triangle | Halfplane]
            triangles split at the given ``point``
        """
        if self._position_of(point) == PointPosition.EDGE_AB:
            return self._split_on_edge(point)
        else:
            return self._split_in_plane(point)

    def _split_in_plane(self, point: tuple[float, float]) -> set[Triangle | Self]:
        """split this half-plane into one Triangle and two half-plane"""
        new_triangle = Triangle(self.point_a, self.point_b, point)
        new_triangles = {new_triangle}
        for halfplane_point in [self.point_a, self.point_b]:
            new_triangles.add(
                Halfplane(
                    halfplane_point,
                    point,
                    new_triangle.outside_point(Straight(halfplane_point, point)),
                )
            )
        return new_triangles

    def _split_on_edge(self, point: tuple[float, float]) -> set[Self]:
        """split this Half-plane at the given ``point`` into two Half-planes"""
        parallel_edge_to_ab = self.edge_ab.parallel_through(self.point_c)

        new_planes = set()
        for halfplane_point in [self.point_a, self.point_b]:
            edge = Straight(halfplane_point, point)
            point_c = parallel_edge_to_ab.point_crossing_with(
                edge.normal_through_middle()
            )
            new_planes.add(Halfplane(halfplane_point, point, point_c))

        return new_planes

    def at_edge(self, point: tuple[float, float]) -> Straight:
        """
        edge where the point is located at

        Parameters
        ----------
        point : tuple[float, float]
            point that is located at an edge of the triangle

        Returns
        -------
        Straight
            edge the given ``point`` lies on
        """
        position = self._position_of(point)
        if position == PointPosition.EDGE_AB:
            return self.edge_ab

    def is_in_circum_circle(
        self, point: tuple[float, float], by: str = "geometry"
    ) -> bool:
        pass

    @cached_property
    def circum_circle_centroid(self) -> tuple[float, float]:
        """compute the circum_centroid of the circum_circle of the edge A-B"""
        return self.point_a[0] + 0.5 * (
            self.point_b[0] - self.point_a[0]
        ), self.point_a[1] + 0.5 * (self.point_b[1] - self.point_a[1])

    @cached_property
    def circum_circle_radius(self) -> float:
        """radius of the circum-circle"""
        return (
            0.5
            * (
                (self.point_b[0] - self.point_a[0]) ** 2.0
                + (self.point_b[1] - self.point_a[1]) ** 2
            )
            ** 0.5
        )

    def is_on_same_side_like(self, other: Self) -> bool:
        """
        check if this and the ``other`` half-plane are on the same side.

        .. versionadded:: 0.1.1

        It is checked if ``point_c`` lies on the other half-plane and
        if the other ``point_c`` lies on this half-plane.

        Parameters
        ----------
        other : Halfplane
            other half-plane to check for the side

        Returns
        -------
        bool
            ``True`` if both are on the same side
        """
        if (
            self._position_of(other.point_c) == PointPosition.INSIDE
            and other._position_of(self.point_c) == PointPosition.INSIDE
        ):
            return True
        else:
            return False
