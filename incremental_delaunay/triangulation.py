"""
triangulation.py

.. versionadded:: 0.1.0

Incremental Delaunay triangulation of a given set of points: 
- DelaunayTriangulationIncrementalWithBoundingBox: starts with a initial bounding
  box that encloses all points and removes the this box after the triangulation is finished
- DelaunayTriangulationIncremental: starts by creating a first triangle by connecting
  three points of the set of points
"""

import operator
from functools import lru_cache, reduce

from .elements import Triangle, Halfplane, are_points_co_linear, Straight, MetaTriangle


class Delaunay:

    """
    Delaunay Triangulation

    .. versionadded:: 0.1.0

    provides basic functionality for a Delaunay triangulation like:
    - finding neighboring triangles to a given triangle
    - flipping triangles when Delaunay conditions is not fulfilled
    - correcting triangles
    - etc.
    """

    __slots__ = "_input_points", "_points", "_triangles", "_super_structure"

    def __init__(self, points: list[tuple[float, float]] | set[tuple[float, float]]):
        """
        Parameters
        ----------
        points : list[tuple[float, float]] | set[tuple[float, float]]
            points that may be triangulated

        Examples
        --------
        To build a mesh of :py:class:`~incremental_delaunay.triangulation.Triangles`
        we first define a list of points.

        >>> point_1 = 0.0, 0.0
        >>> point_2 = 0.0, 1.0
        >>> point_3 = 1.0, 0.0
        >>> input_points = [point_1, point_2, point_3]

        The list of points is simply passed to
        :py:class:`~incremental_delaunay.triangulation.DelaunayTriangulation`.

        >>> from incremental_delaunay.triangulation import Delaunay
        >>> delaunay = Delaunay(input_points)

        During initilization the mesh is triangulated.
        The computed triangles are stored in the attribute
        :py:attr:`~incremental_delaunay.DelaunayTriangulation.triangles`.

        >>> delaunay.triangles
        [Triangle(point_a=(0.0, 0.0), point_B=(0.0, 1.0), point_C=(1.0, 0.0))]

        If we want to check, in what triangle an arbritrary point is located
        we must only iterate over the computed :py:class:`~incremental_delaunay.triangulation.Triangle`.

        >>> arbitrary_point = 0.2, 0.2
        >>> for triangle in delaunay.triangles:
        ...    if triangle.is_inside(arbitrary_point):
        ...       return triangle
        Triangle(point_a=(0.0, 0.0), point_B=(0.0, 1.0), point_C=(1.0, 0.0))

        In case the ``arbitrary_point`` is outside the computed mesh, the
        above given for-loop returns ``None``.
        """
        self._input_points = set(points)
        self._points = set()
        self._triangles = set()

    def __repr__(self) -> str:
        return f"Delaunay(points={self._input_points})"

    @property
    def points(self) -> set[tuple]:
        """added points"""
        return self._points

    @property
    def triangles(self) -> set[Triangle]:
        """triangles constructed from the points"""
        return self._triangles

    def _compare_input_and_triangle_points(self) -> None:
        """
        check if input-points and points in the triangles are the same

        Raises
        ------
        ValueError
            If input-points are missing in the mesh / triangles
        ValueError
            If triangle-points are not part of the mesh
        """
        triangle_points = set()
        for triangle in self.triangles:
            triangle_points.update(set(triangle.points))
        missing_points = self._input_points.difference(triangle_points)
        if len(missing_points) > 0:
            for triangle in self.triangles:
                print(f"{triangle.points},")
            for point in missing_points:
                self.add(point)
        to_much_points = triangle_points.difference(self._input_points)
        if len(to_much_points) > 0:
            raise ValueError(
                f"Triangle points {to_much_points} are not give in the input-points"
            )

    def add(self, point: tuple[float, float]) -> None:
        """add point to given mesh"""
        ...

    def _replace_triangles(self, add, remove) -> None:
        """replace triangle"""
        ...

    def _point_is_outside_mesh(self, point: tuple[float, float]) -> bool:
        """check if given ``point`` is outside the computed mesh"""
        for triangle in self.triangles:
            if isinstance(triangle, Triangle) and triangle.is_inside(point):
                return False
        return True

    def _split_neighbouring_triangle(
        self, point: tuple[float, float], triangle: Triangle
    ) -> None:
        """
        split the given ``triangle`` at the given ``point``

        Parameters
        ----------
        point : tuple[float, float]
            point where the ``triangle`` is to be split
        triangle : :py:class:`~incremental_delaunay.triangulation.Triangle`
            triangle to be split

        Returns
        -------
        None
        """
        edge = triangle.at_edge(point)
        neighbouring_triangles = self._neighboring_triangles(triangle)
        for neighbouring_triangle in neighbouring_triangles:
            if edge in neighbouring_triangle.edges:
                self._replace_triangles(
                    add=neighbouring_triangle.split(point),
                    remove={neighbouring_triangle},
                )

    def _correct_triangles(self) -> bool:
        """
        check all triangles if correction is needed

        Returns
        -------
        bool
            ``True`` if triangles have been corrected.
            ``False`` if no correction was needed.
        """
        for triangle in self.triangles:
            if not isinstance(triangle, Triangle):
                continue
            neighbouring_triangles = self._neighboring_triangles(triangle)
            for neighboring_triangle in neighbouring_triangles:
                if not isinstance(neighboring_triangle, Triangle):
                    continue
                not_shared_point = neighboring_triangle.not_shared_point(triangle)
                if triangle.is_in_circum_circle(not_shared_point):
                    self.flip(triangle, neighboring_triangle)
                    return True
        return False

    def _neighboring_triangles(self, triangle: Triangle) -> set[Triangle]:
        """
        determine the neighbouring triangles to the given ``triangle``

        Parameters
        ----------
        triangle : Triangle
            triangle which neighbouring triangles are searched

        Returns
        -------
        list[Triangle]
            the neighbouring triangles to the given one
        """
        neighbours = set()
        for neighbour in self.triangles:
            if neighbour.shares_edge_with(triangle) and neighbour != triangle:
                neighbours.add(neighbour)
        return neighbours

    def flip(self, triangle_1: Triangle, triangle_2: Triangle) -> None:
        """
        flip the both given triangles and remove them
        from list of triangles

        Both triangles must share one edge
        Furthermore, it is assumed that one point of one of the given triangles
        does not fulfill the Delaunay-condition.
        I.e. is within the circum-circle of the other triangle.

        Parameters
        ----------
        triangle_1 : Triangle
            first triangle
        triangle_2 : Triangle
            second triangle

        Returns
        -------
        None
        """
        if not isinstance(triangle_1, Triangle) or not isinstance(triangle_2, Triangle):
            return
        not_shared_points = triangle_1.not_shared_points(triangle_2)
        shared_points = triangle_1.shared_edge(triangle_2)
        if not are_points_co_linear(
            not_shared_points[0], not_shared_points[1], shared_points[0]
        ) and not are_points_co_linear(
            not_shared_points[0], not_shared_points[1], shared_points[1]
        ):
            new_triangles = {
                Triangle(point, not_shared_points[0], not_shared_points[1])
                for point in shared_points
            }
            old_triangles = {triangle_1, triangle_2}
            self._replace_triangles(add=new_triangles, remove=old_triangles)


class DelaunayTriangulationIncrementalWithBoundingBox(Delaunay):

    """
    Delaunay-triangulation with bounding box

    .. versionadded:: 0.1.0

    Conducts an incremental Delaunay-triangulation starting with an initial bounding
    structure like a triangle or a rectangle, that encloses all points and removes
    the this box after the triangulation is finished
    """

    def __init__(
        self, points: list[tuple[float, float]], bounding_structure="rectangle"
    ):
        """
        Parameters
        ----------
        points : list[tuple[float, float]]
            points that may be triangulated
        bounding_structure : str
            choose what superstructure to use.
            Possible values are: ``'triangle'`` and ``'rectangle'`` (Default)

        Examples
        --------
        To build a mesh of :py:class:`~incremental_delaunay.triangulation.Triangles`
        we first define a list of points.

        >>> point_1 = 0.0, 0.0
        >>> point_2 = 0.0, 1.0
        >>> point_3 = 1.0, 0.0
        >>> input_points = [point_1, point_2, point_3]

        The list of points is simply passed to
        :py:class:`~incremental_delaunay.triangulation.DelaunayTriangulation`.

        >>> from incremental_delaunay.triangulation import DelaunayTriangulationIncrementalWithBoundingBox
        >>> delaunay = DelaunayTriangulationIncrementalWithBoundingBox(input_points)

        During initilization the mesh is triangulated.
        The computed triangles are stored in the attribute
        :py:attr:`~incremental_delaunay.DelaunayTriangulation.triangles`.

        >>> delaunay.triangles
        [Triangle(point_a=(0.0, 0.0), point_B=(0.0, 1.0), point_C=(1.0, 0.0))]

        If we want to check, in what triangle an arbritrary point is located
        we must only iterate over the computed :py:class:`~incremental_delaunay.triangulation.Triangle`.

        >>> arbitrary_point = 0.2, 0.2
        >>> for triangle in delaunay.triangles:
        ...    if triangle.is_inside(arbitrary_point):
        ...       return triangle
        Triangle(point_a=(0.0, 0.0), point_B=(0.0, 1.0), point_C=(1.0, 0.0))

        In case the ``arbitrary_point`` is outside the computed mesh, the
        above given for-loop returns ``None``.
        """
        self._bounding_structure = bounding_structure
        super().__init__(points)
        self._triangulation()
        self._compare_input_and_triangle_points()

    @property
    def bounding_structure(self) -> str:
        """bounding-structure for the triangulation"""
        return self._bounding_structure

    def add(self, point: tuple[float, float]) -> None:
        """
        Adds the given ``point`` to the triangulation.

        After ``point`` is added all triangles are checked for fulfilling the
        Delaunay-condition.

        Parameters
        ----------
        point : tuple[float, float]
            point with two coordinates

        Returns
        -------
        None
        """
        for triangle in self.triangles:
            if point in triangle.points:
                continue
            if triangle.is_inside(point):
                if triangle.is_on_edge(point):
                    self._split_neighbouring_triangle(point, triangle)
                new_triangles = triangle.split(point)
                self._replace_triangles(add=new_triangles, remove={triangle})
                self._points.add(point)
                for _ in range(len(self.triangles)):
                    if not self._correct_triangles():
                        break
                return
        print(f"{point=} is outside mesh")

    @staticmethod
    def _sort_points_along_edges(
        border_edges: list[Straight],
    ) -> list[tuple[float, float]]:
        """
        sort the points of the ``border_edges`` around the mesh

        goes through the list of ``border_edges`` and compares if
        one point of the ``border_edge`` is equal to the last sorted
        point and the other point of the ``border_edge`` is not the
        second last sorted point.
        If this applies the other point of the ``border_edge`` (see above)
        is added.

        Parameters
        ----------
        border_edges: list[Straight]
            all edges around the computed mesh

        Returns
        -------
        list[tuple[float, float]]
            all points of the edges sorted next to each other
        """
        sorted_points = [border_edges[0].point_1, border_edges[0].point_2]
        for _ in range(len(border_edges) + 1):
            for border_edge in border_edges:
                if (
                    border_edge.point_1 == sorted_points[-1]
                    and border_edge.point_1 != sorted_points[-2]
                ):
                    sorted_points.append(border_edge.point_2)
                    break
                elif (
                    border_edge.point_2 == sorted_points[-1]
                    and border_edge.point_1 != sorted_points[-2]
                ):
                    sorted_points.append(border_edge.point_1)
                    break

        for border_edge in border_edges:
            edge_exists = False
            for index in range(len(sorted_points) - 1):
                if (
                    border_edge.point_1 == sorted_points[index]
                    and border_edge.point_2 == sorted_points[index + 1]
                ) or (
                    border_edge.point_2 == sorted_points[index]
                    and border_edge.point_1 == sorted_points[index + 1]
                ):
                    edge_exists = True
                    break  # edge exist in sorted points
            if not edge_exists:
                print(f"{border_edge=} is missing in {sorted_points=}")
        return sorted_points

    def _border_edges(self, border_triangle: Triangle) -> list[Straight]:
        """get the border-edges of the given ``border_triangle``"""
        neighbours = self._neighboring_triangles(border_triangle)
        edges = list(border_triangle.edges)
        for neighbour in neighbours:
            for edge in border_triangle.edges:
                if neighbour.is_edge(edge):
                    edges.remove(edge)
        return edges

    def _is_border_triangle(self, triangle: Triangle) -> bool:
        """
        check if triangle is at the border of triangulated mesh

        assumes that a triangle at the border of a triangulated mesh
        has less than three neighboring triangles

        Parameters
        ----------
        triangle : Triangle
            instance to be checked for neighbours

        Returns
        -------
        bool
            ``True`` if ``triangle`` has less than three neighboring triangles.
            ``False`` if ``triangle`` has at least three neighbors.
        """
        if len(self._neighboring_triangles(triangle)) < 3:
            return True
        else:
            return False

    def _triangulation(self) -> None:
        """
        performs a full triangulation with the points given by the user
        at initialization
        """
        if len(self.triangles) == 0:
            self._initialize_super_structure()

        for point in self._input_points:
            self.add(point)
        self._remove_super_structure()
        self._check_and_update_border_triangles()

    def _replace_triangles(self, add: set[Triangle], remove: set[Triangle]) -> None:
        """
        add and remove the given triangles from :py:class:`~incremental_delaunay.triangulation.Triangle.triangles`

        Parameters
        ----------
        add : set[Triangle]
            :py:class:`~incremental_delaunay.triangulation.Triangle` to be added to
            :py:class:`~incremental_delaunay.triangulation.Triangle.triangles`
        remove : set[Triangle]
            :py:class:`~incremental_delaunay.triangulation.Triangle` to be removed from
            :py:class:`~incremental_delaunay.triangulation.Triangle.triangles`

        Returns
        -------
        None
        """
        for triangle in remove:
            self._triangles.remove(triangle)
        self._triangles = self.triangles.union(add)

    def _check_and_update_border_triangles(self):
        """
        check the borders of the triangle and update it
        in case of a concave / non-convex form

        First the triangles at the border are determined (border-triangles).
        Afterwards, the edges of these border-triangles that are exposed
        to the outside are determined.
        To get the outside shell of the mesh, the points of the edges are
        sorted after each other to a circle.
        Then, three points at a time are checked for their form.
        In case these points form a bump / are concave compared to the
        mesh, a new triangle is added between them.
        """
        border_triangles = [
            triangle
            for triangle in self.triangles
            if self._is_border_triangle(triangle)
        ]
        border_edges = reduce(
            operator.add,
            [self._border_edges(triangle) for triangle in border_triangles],
        )
        sorted_border_points = self._sort_points_along_edges(border_edges)
        self._add_border_triangles(sorted_border_points)
        self._correct_triangles()

    def _add_border_triangles(self, border_points: list[tuple[float, float]]) -> None:
        """
        add missing triangles between the given ``border_points``
        if they form a concave shell of the mesh's border

        Parameters
        ----------
        border_points : list[tuple[float, float]]
            points at the border sorted in a row to a closed circle

        Returns
        -------
        None
        """
        border_points.append(border_points[1])
        skip_index = -1
        for index in range(len(border_points) - 2):
            if index == skip_index:
                continue
            m_point = Straight(
                border_points[index], border_points[index + 2]
            ).middle_point
            if self._point_is_outside_mesh(m_point) and not are_points_co_linear(
                border_points[index], border_points[index + 1], border_points[index + 2]
            ):
                new_triangle = Triangle(
                    border_points[index],
                    border_points[index + 1],
                    border_points[index + 2],
                )
                self._triangles.add(new_triangle)
                skip_index = index + 1
        self._correct_triangles()

    def _initialize_super_structure(self) -> None:
        """
        Initialize a super-structure that is deleted after the mesh is created
        """
        if self.bounding_structure.lower() == "triangle":
            self._triangles.add(self._super_triangle())
        elif self.bounding_structure.lower() == "rectangle":
            self._triangles.update(self._super_rectangle())
        else:
            self._triangles.update(self._super_rectangle())

    def _remove_super_structure(self) -> None:
        """
        remove the super-structure (triangle / rectangle)
        Depends on user-input of argument ``bounding_structure``
        """
        for triangle in self.triangles:
            print(f"{triangle.points}, ")
        print("")
        if self.bounding_structure.lower() == "triangle":
            self._remove_super_triangle()
        elif self.bounding_structure.lower() == "rectangle":
            self._remove_super_rectangle()
        else:
            self._remove_super_rectangle()

    def _remove_super_triangle(self) -> None:
        """
        removes all triangles that share at least
        one point with the super-triangle
        """
        super_triangle = self._super_triangle()
        cleared_triangles = {
            triangle
            for triangle in self.triangles
            if not triangle.shares_point(super_triangle)
        }
        self._triangles = cleared_triangles

    def _remove_super_rectangle(self) -> None:
        """
        removes all triangles that share at least one
        point with the super-rectangle
        """
        super_rectangle = self._super_rectangle()
        super_rectangle_points = set()
        for triangle in super_rectangle:
            super_rectangle_points.update(set(triangle.points))
        cleared_triangles = set()
        for triangle in self.triangles.copy():
            if triangle.shares_point(super_rectangle_points):
                continue
            cleared_triangles.add(triangle)
        self._triangles = cleared_triangles

    @lru_cache
    def _super_triangle(self) -> Triangle:
        """create a super-triangle that contains all points in input_points"""
        x, y = self.max_points()
        x_distance = x[1] - x[0]
        y_distance = y[1] - y[0]
        point_a = x[0] + 2.5 * x_distance, y[0] + 0.5 * y_distance
        point_b = x[0] + 0.5 * x_distance, y[0] + 2.5 * y_distance
        point_c = x[0] - 2.0 * x_distance, y[0] - 2.0 * y_distance
        super_triangle = Triangle(point_a, point_b, point_c)
        return super_triangle

    @lru_cache
    def _super_rectangle(self) -> set[Triangle, Triangle]:
        """create a super-triangle that contains all points in ``input_points``"""
        x, y = self.max_points()
        x_mean = 0.5 * sum(x)
        y_mean = 0.5 * sum(y)
        factor = 0.9
        x_distance = (x[1] - x[0]) * factor
        y_distance = (y[1] - y[0]) * factor
        point_a = x_mean - x_distance, y_mean - y_distance
        point_b = x_mean - x_distance, y_mean + y_distance
        point_c = x_mean + x_distance, y_mean + y_distance
        point_d = x_mean + x_distance, y_mean - y_distance
        triangle_1 = Triangle(point_a, point_b, point_c)
        triangle_2 = Triangle(point_a, point_c, point_d)
        return {triangle_1, triangle_2}

    def max_points(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Initial guess of the maximum points of the given initial point-set

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            X(min, max), Y(min, max)
        """
        min_x = min(self._input_points, key=operator.itemgetter(0))[0]
        min_y = min(self._input_points, key=operator.itemgetter(1))[1]
        max_x = max(self._input_points, key=operator.itemgetter(0))[0]
        max_y = max(self._input_points, key=operator.itemgetter(1))[1]
        return (min_x, max_x), (min_y, max_y)


class DelaunayTriangulationIncremental(Delaunay):

    """
    Incremental Delaunay Triangulation

    .. versionadded:: 0.1.0
    """

    def __init__(self, points: list[tuple[float, float]]):
        """
        Parameters
        ----------
        points : list[tuple[float, float]]
            points that may be triangulated

        Examples
        --------
        To build a mesh of :py:class:`~incremental_delaunay.triangulation.Triangles`
        we first define a list of points.

        >>> point_1 = 0.0, 0.0
        >>> point_2 = 0.0, 1.0
        >>> point_3 = 1.0, 0.0
        >>> input_points = [point_1, point_2, point_3]

        The list of points is simply passed to
        :py:class:`~incremental_delaunay.triangulation.DelaunayTriangulation`.

        >>> from incremental_delaunay.triangulation import DelaunayTriangulationIncremental
        >>> delaunay = DelaunayTriangulationIncremental(input_points)

        During initilization the mesh is triangulated.
        The computed triangles are stored in the attribute
        :py:attr:`~incremental_delaunay.DelaunayTriangulation.triangles`.

        >>> delaunay.triangles
        [Triangle(point_a=(0.0, 0.0), point_B=(0.0, 1.0), point_C=(1.0, 0.0))]

        If we want to check, in what triangle an arbritrary point is located
        we must only iterate over the computed :py:class:`~incremental_delaunay.triangulation.Triangle`.

        >>> arbitrary_point = 0.2, 0.2
        >>> for triangle in delaunay.triangles:
        ...    if triangle.is_inside(arbitrary_point):
        ...       return triangle
        Triangle(point_a=(0.0, 0.0), point_B=(0.0, 1.0), point_C=(1.0, 0.0))

        In case the ``arbitrary_point`` is outside the computed mesh, the
        above given for-loop returns ``None``.
        """
        super().__init__(points)
        self._halfplanes: set[Halfplane] = set()
        self._triangulation()
        self._compare_input_and_triangle_points()

    def __repr__(self) -> str:
        return f"DelaunayTriangulationIncremental(points={self._input_points})"

    def _triangle_points(self) -> set[tuple[float, float]]:
        """
        collect all points of the triangles

        Returns
        -------
        set[tuple[float, float]]
            points of all triangles
        """
        points = set()
        for triangle in self.triangles:
            points.update(set(triangle.points))
        return points

    def add(self, point: tuple[float, float]) -> None:
        """
        Adds the given ``point`` to the triangulation.

        After ``point`` is added all triangles are checked for fulfilling the
        Delaunay-condition.

        Parameters
        ----------
        point : tuple[float, float]
            point with two coordinates

        Returns
        -------
        None
        """
        if point in self.points:
            return
        self._points.add(point)
        added_triangles = set()
        remove = set()
        for triangle in self._all_triangles():
            if triangle.is_inside(point):
                if triangle.is_on_edge(point):
                    self._split_neighbouring_triangle(point, triangle)
                added_triangles.update(triangle.split(point))
                remove.update({triangle})
                break
        self._replace_triangles(added_triangles, remove)

        # add triangles where concave hull is available
        while self._add_triangle_at_concave_hull():
            pass

        # correct triangles if delaunay-condition is not fulfilled
        self._correct_triangles()

    def _add_triangle_at_concave_hull(
        self,
    ) -> bool:
        """
        add a triangle and corresponding half-planes where
        the outside border is concave

        'concave' means the border of the mesh is curved inward.
        The opposite of concave is convex.

        Returns
        -------
        bool
            ``True`` if convex triangle has been added.
            ``False`` if no triangle has been added.
        """
        for halfplane_1 in self._halfplanes:
            for halfplane_2 in self._halfplanes:
                if halfplane_1.is_neighbour(
                    halfplane_2
                ) and halfplane_1.is_on_same_side_like(halfplane_2):
                    not_shared_points = halfplane_1.edge_ab.not_shared_points(
                        halfplane_2.edge_ab
                    )
                    line = Straight(not_shared_points[0], not_shared_points[1])
                    if self._point_is_outside_mesh(line.middle_point):
                        # add Triangle
                        convex_triangle = Triangle(
                            halfplane_1.edge_ab.shared_point(halfplane_2.edge_ab),
                            not_shared_points[0],
                            not_shared_points[1],
                        )

                        # add corresponding Halfplane
                        convex_halfplane = Halfplane(
                            not_shared_points[0],
                            not_shared_points[1],
                            convex_triangle.outside_point(line),
                        )
                        self._replace_triangles(
                            add={convex_triangle, convex_halfplane},
                            remove={halfplane_1, halfplane_2},
                        )
                        return True
        return False

    def _add_first_triangle(self) -> None:
        """
        builds the first triangle and adds a half-plane
        on each edge
        """
        points = self._find_non_colinear_points()
        new_triangle = Triangle(points[0], points[1], points[2])
        self._points.update(set(points))
        new_triangles = {new_triangle}
        for edge in new_triangle.edges:
            new_triangles.add(
                Halfplane(edge.point_1, edge.point_2, new_triangle.outside_point(edge))
            )
        self._replace_triangles(add=new_triangles, remove=set())

    def _find_non_colinear_points(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        find three points that are not co-linear

        These points are needed to build the first triangle.
        The three points must not lie on one line.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
            three points that are not co-linear
        """
        points = list(self._input_points)
        for index in range(len(points) - 2):
            if not are_points_co_linear(
                points[index], points[index + 1], points[index + 2]
            ):
                return points[index], points[index + 1], points[index + 2]

    def _triangulation(self) -> None:
        """triangulate by adding one point after another"""
        self._add_first_triangle()
        for point in self._input_points:
            self.add(point)

    def _all_triangles(self) -> set[Triangle, Halfplane]:
        """
        returns a set all triangles and half-planes in one list

        Returns
        -------
        set[Triangle, Halfplane]
            all triangulated triangles and half-planes
        """
        all_triangles = set()
        all_triangles.update(self._triangles)
        all_triangles.update(self._halfplanes)
        return all_triangles

    def _replace_triangles(
        self,
        add: set[Triangle | Halfplane],
        remove: set[Triangle | Halfplane],
    ) -> None:
        """
        replace the ``remove`` instances with the ``add`` instances
        """
        for triangle in remove:
            if isinstance(triangle, Triangle):
                self._triangles.remove(triangle)
            else:
                self._halfplanes.remove(triangle)
        for triangle in add:
            if isinstance(triangle, Triangle):
                self._triangles.add(triangle)
            else:
                self._halfplanes.add(triangle)

    def _neighboring_triangles(self, triangle: Triangle) -> set[Triangle]:
        """
        determine the neighbouring triangles to the given ``triangle``

        Parameters
        ----------
        triangle : Triangle
            triangle which neighbouring triangles are searched

        Returns
        -------
        list[Triangle]
            the neighbouring triangles to the given one
        """
        neighbours = set()
        for neighbour in self._all_triangles():
            if neighbour.shares_edge_with(triangle) and neighbour != triangle:
                neighbours.add(neighbour)
        return neighbours
