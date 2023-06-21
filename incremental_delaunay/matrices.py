"""
matrices.py

.. versionadded:: 0.1.0 

implementation of basic matrix-operations
"""
import operator
from functools import reduce


class Vector:

    """
    Vector

    .. versionadded:: 0.1.0
    """

    __slots__ = "_entries", "_entry_iterator"

    def __init__(self, entries: list):
        """
        Parameters
        ----------
        entries : list
            entries of the vector

        See Also
        --------
        Matrix : matrix with basic functionality of matrices-computation

        Examples
        --------
        An instance of :py:class:`~m_n_kappa.matrices.Vector` is easily
        initialized by passing a list of ``float``

        >>> from m_n_kappa.matrices import Vector
        >>> x = Vector([0, 1])
        >>> x
        Vector([0.0, 1.0])

        Adding another :py:class:`~m_n_kappa.matrices.Vector` is done by using the
        plus (``+``) operator.
        Please be aware that both vectors need to be of the same length.

        >>> y = Vector([2, 3])
        >>> x + y
        Vector([2.0, 4.0])

        Alternatively you may use the :py:meth:`~m_n_kappa.matrices.Vector.add`-method.

        >>> x.add(y)
        Vector([2.0, 4.0])

        Subtracting an :py:class:`~m_n_kappa.matrices.Vector` is also done like in two ways.
        By the Minus-Operator (``-``).

        >>> x - y
        Vector([-2.0, -2.0])

        Or by using the :py:meth:`~m_n_kappa.matrices.Vector.subtract`-method.

        >>> x.subtract(y)
        Vector([-2.0, -2.0])

        If you want to compute the `scalar-product <https://en.wikipedia.org/wiki/Dot_product>`_
        (also called dot-product, :math:`x^{T} y`) you may use the
        :py:meth:`~m_n_kappa.matrices.Vector.scalar_product`-method.

        >>> x.scalar_product(y)
        3.0

        Multiplying a scalar is done using the :py:meth:`~m_n_kappa.matrices.Vector.multiply_scalar`-method.

        >>> x.multiply_scalar(2.0)
        Vector([0.0, 2.0])

        The :py:meth:`~m_n_kappa.matrices.Vector.tensor_product`-method therefore creates a
        :py:class:`~m_n_kappa.matrices.Matrix`.

        >>> x.tensor_product(y)
        Matrix([[0.0, 0.0], [2.0, 3.0]])

        If you want to modify a value of your Vector you may use the
        :py:meth:`~m_n_kappa.matrices.Vector.replace`-method ``inplace``.

        >>> x.replace(number=0, value=2.0, inplace=True)
        >>> x
        Vector([2.0, 2.0])

        Or not ``inplace`` getting another :py:class:`~m_n_kappa.matrices.Vector`
        in return.

        >>> x.replace(number=0, value=2.0)
        Vector([2.0, 2.0])

        Using a :py:class:`~m_n_kappa.matrices.Vector` in a ``for``-loop is easily done
        as follows.

        >>> for entry in x:
        ...     print(entry)
        0
        2

        Comparing two vectors is also easy.

        >>> x == y
        False

        Whereas

        >>> x == x
        True

        """
        self._entries = entries

    def __repr__(self) -> str:
        return f"Vector({self._entries})"

    def __iter__(self):
        self._entry_iterator = iter(self.entries)
        return self

    def __next__(self):
        return self._entry_iterator.__next__()

    def __len__(self) -> int:
        return len(self.entries)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vector) and self._entries == other._entries:
            return True
        else:
            return False

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __round__(self, n=None):
        rounded_entries = [round(entry, n) for entry in self.entries]
        return Vector(rounded_entries)

    @property
    def entries(self) -> list:
        """entries of the vector"""
        return self._entries

    def length(self) -> float:
        """sum of the entries of this vector"""
        return sum(self.entries)

    def replace(self, number: int, value: float, inplace: bool = False) -> None:
        """
        replace the entry of the given ``number`` with ``value``

        Parameters
        ----------
        number : int
            entry-number to replace
        value : float
            replace-value
        inplace : bool
            If ``True`` changes value in place and does not return a value.
            If ``False`` returns this Vector with ``value`` at the desired position
            (``number``, Default: ``False``).

        Returns
        -------
        None | Vector
        """
        if inplace:
            self._entries[number] = float(value)
            return
        else:
            new_vector = Vector(self.entries[:])
            new_vector._entries[number] = float(value)
            return new_vector

    def append(self, vector):
        """
        Append another vector to this vector

        Both vectors need to have the same length

        Parameters
        ----------
        vector : Vector
            vector to append to this vector

        Returns
        -------
        Matrix
            matrix from this and the appended vector

        Examples
        --------
        >>> from m_n_kappa.matrices import Vector
        >>> vector_1 = Vector([0, 1])
        >>> vector_2 = Vector([2, 3])
        >>> vector_1.append(vector_2)
        Matrix([[0.0, 2.0], [1.0, 3.0]])
        """
        self._has_same_length(vector)
        matrix = []
        for index, entry in enumerate(vector):
            matrix.append([self.entries[index], entry])
        return Matrix(matrix)

    def tensor_product(self, vector):
        """
        compute the tensor-product of this vector with another vector

        Parameters
        ----------
        vector : Vector
            other vector to compute the tensor-product with

        Returns
        -------
        Matrix
            m x n - matrix.
            Where m is the number of rows that is equal to the number of entries.
            And n is the number of columns that is equal to the number of entries of the vector.
        """
        matrix = []
        for entry in self.entries:
            matrix.append([entry * other_entry for other_entry in vector])
        return Matrix(matrix)

    def scalar_product(self, vector) -> float:
        """
        compute the scalar-product of this vector with another vector

        Parameters
        ----------
        vector : Vector
            other vector to compute the tensor-product with

        Returns
        -------
        float
            scalar-product of this vector with another vector

        Examples
        --------
        >>> from m_n_kappa.matrices import Vector
        >>> vector_1 = Vector([0, 1])
        >>> vector_2 = Vector([2, 3])
        >>> vector_1.scalar_product(vector_2)
        3.0
        """
        return sum((self.entries[index] * entry for index, entry in enumerate(vector)))

    def multiply_scalar(self, scalar: float):
        """
        multiplies a scalar to this vector

        Parameters
        ----------
        scalar : float
            number to multiply with this vector

        Returns
        -------
        Vector
            vector where each entry is multiplied with the given number / scalar
        """
        return Vector([scalar * entry for entry in self.entries])

    def norm(self) -> float:
        """
        Euclidian norm of the vector
        """
        return (sum((entry * entry for entry in self._entries))) ** 0.5

    def mean(self) -> float:
        """
        Mean of the values in the vector

        Notes
        -----

        .. math::

           \\overline{x} = \\frac{\\sum_{i=1}^{n} x_i}{n}
        """
        return sum(self.entries) / len(self.entries)

    def least_squares(self) -> float:
        """
        compute the least-squares of the vector

        Notes
        -----

        .. math::

               \\sum_{i = 0}^{n} (x_i - \\overline{x})^{2}

        where :math:`\\overline{x}` is the mean of the values in the vector.
        """
        mean = self.mean()
        return sum(((entry - mean) ** 2.0 for entry in self.entries))

    @staticmethod
    def _check_is_vector(vector):
        """check if argument is a vector"""
        if not isinstance(vector, Vector):
            TypeError(f"Vector is not of type 'Vector'. Is of type {type(vector)}.")

    def add(self, vector):
        """
        Add this vector to another

        Parameters
        ----------
        vector : Vector
            other vector

        Returns
        -------
        Vector
            Sum of both vector (this one and the other one)
        """
        self._check_is_vector(vector)
        if self.__len__() != len(vector):
            ValueError("Both vectors must have the same length.")
        return Vector(
            [self.entries[index] + entry for index, entry in enumerate(vector)]
        )

    def subtract(self, vector):
        """
        Subtract ``vector`` from this vector

        Parameters
        ----------
        vector : Vector
            vector to subtract from this vector

        Returns
        -------
        Vector
            result of the subtraction
        """
        subtract_vector = vector.multiply_scalar(-1.0)
        return self.add(subtract_vector)

    def _has_same_length(self, vector) -> None:
        """check if this and the other vector have the same length"""
        if self.__len__() != len(vector):
            raise ValueError("This and the other vector need to have the same length.")


class Matrix:

    """
    Matrix

    .. versionadded:: 0.1.0

    implement basic functionality of matrix computation
    """

    __slots__ = "_matrix"

    def __init__(self, matrix: list[list]):
        """
        Parameters
        ----------
        matrix : list[list]
            Representation of the matrix.
            Each entry of the first outer list represents a row.
            Each entry of the inner list represents the value.

        See Also
        --------
        Vector : class for vector-operations

        Examples
        --------
        A :py:class:`~m_n_kappa.matrices.Matrix` is initialized by passing
        a list of lists.
        The inner list represents a row, wheras the outer list represents the hole matrix.

        >>> from m_n_kappa.matrices import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> A
        Matrix([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

        You can transpose the matrix ``A`` by using the :py:meth:`~m_n_kappa.matrices.Matrix.transpose`-
        method.

        >>> A.transpose()
        Matrix([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])

        Several multiplication-operations are given using :py:meth:`~m_n_kappa.matrices.Matrix.multiply_by`.
        For example multiplying a ``float``.

        >>> A.multiply_by(5)
        Matrix([[0.0, 5.0, 10.0], [15.0, 20.0, 25.0]])

        A :py:class:`~m_n_kappa.matrices.Vector`, what itself leads to a new Vector.

        >>> A.multiply_by(Vector([1, 2, 3]))
        Vector([8.0, 26.0])

        Or another :py:class:`~m_n_kappa.matrices.Matrix`:

        >>> B = Matrix([[1, 2], [0, 1], [4, 0]])
        >>> A.multiply_by(B)
        Matrix([[9.0, 1.0], [23.0, 10.0]])

        Adding and subtracting are done by using the corresponding methods
        :py:meth:`~m_n_kappa.matrices.Matrix.add` and :py:meth:`~m_n_kappa.matrices.Matrix.subtract`.

        >>> A.add(Matrix([6, 7, 8], [9, 10, 11]))
        Matrix([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])

        >>> A.subtract(Matrix([6, 7, 8], [9, 10, 11]))
        Matrix([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]])

        As an alternative you may of course use the corresponding operators (``+`` and ``-``).

        >>> A + Matrix([6, 7, 8], [9, 10, 11])
        Matrix([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]])

        >>> A - Matrix([6, 7, 8], [9, 10, 11])
        Matrix([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]])

        In case you want a specific row or a specific column given as a :py:class:`~m_n_kappa.matrices.Vector`
        you only have to use the :py:meth:`~m_n_kappa.matrices.Matrix.row_vector`- or the
        :py:meth:`~m_n_kappa.matrices.Matrix.column_vector`-method.
        The number passed to the method describes the wanted row or column.

        >>> A.column_vector(1)
        Vector([1.0, 4.0]]

        >>> A.row_vector(1)
        Vector([3.0, 4.0, 5.0])

        .. note::

           Please be aware that python starts counting at zero (``0``).
           Therefore, if you want the first row of A (``=[0, 1, 2]``) you need to pass 0
           to :py:meth:`~m_n_kappa.matrices.Matrix.row_vector`.
           Same applies to :py:meth:`~m_n_kappa.matrices.Matrix.column_vector`.

        Furthermore, :py:class:`~m_n_kappa.matrices.Matrix` supports
        `QR decomposition <https://en.wikipedia.org/wiki/QR_decomposition>`, that
        determines from the given matrix an orthonormalized matrix and an upper
        triangular matrix using the :py:meth:`~m_n_kappa.matrices.Matrixorthonormal_triangular`-
        method.

        >>> QR =  Matrix([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
        >>> q, t = QR.orthonormal_triangular(algorithm='Modified Gram-Schmidt')

        In this example ``q`` is the orthonormalized matrix.

        >>> q
        Matrix([[6 / 7, -69 / 175, -58 / 175],[3 / 7, 158 / 175, 6 / 175],[-2 / 7, 6 / 35, -33 / 35]])

        and ``t`` the triangular matrix.
        >>> t
        Matrix([[14, 21, -14], [0, 175, -70], [0, 0, 35]])
        """
        self._matrix = self._floatify(matrix)

    def __repr__(self) -> str:
        return f"Matrix({self.matrix})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Matrix) and self._matrix == other.matrix:
            return True
        else:
            return False

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __mul__(self, other):
        return self.multiply_by(other)

    def __round__(self, n=None):
        rounded_matrix = [[round(entry, n) for entry in row] for row in self.matrix]
        return Matrix(rounded_matrix)

    @property
    def matrix(self) -> list[list]:
        """shows the matrix in ``list``-form"""
        return self._matrix

    @property
    def row_number(self) -> int:
        """gives the number of rows of the matrix"""
        return len(self.matrix)

    @property
    def column_number(self) -> int:
        """gives number of columns of the matrix"""
        return len(self.matrix[0])

    def _floatify(self, matrix: list[list]) -> list[list]:
        """makes sure that each entry is of type float"""
        new_matrix = [list(map(float, row)) for row in matrix]
        return new_matrix

    def append(self, vector: Vector):
        """
        Appends the given vector as column-vector to this vector

        Parameters
        ----------
        vector : Vector
            column-vector to append to this matrix

        Returns
        -------
        Matrix
            this matrix with appended vector as additional column-vector
        """
        if not self.matrix:
            return Matrix([[entry] for entry in vector.entries])
        matrix = self.matrix
        for index, entry in enumerate(vector):
            matrix[index].append(entry)
        return Matrix(matrix)

    def column(self, number: int) -> list:
        """
        give the column with the given number

        Parameters
        ----------
        number : int
            number of the needed column

        Returns
        -------
        list
            values of the given column in a ``list`` from top to bottom
        """
        return [row[number] for row in self.matrix]

    def row(self, number: int) -> list:
        """
        give the row with the given number

        Parameters
        ----------
        number : int
            number of the needed row

        Returns
        -------
        list
            values of the needed row
        """
        return self.matrix[number]

    def transpose(self):
        """
        transposes the matrix and gives a new matrix

        Examples
        --------
        >>> from m_n_kappa.matrices import Matrix
        >>> a_matrix = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> a_matrix.transpose()
        Matrix([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
        """
        matrix = []
        for row_number, row in enumerate(self.matrix):
            for column_number, entry in enumerate(row):
                if len(matrix) == column_number:
                    matrix.append([])
                matrix[column_number].append(entry)
        return Matrix(matrix)

    def entry(self, row_number: int, column_number: int) -> float:
        """
        get the of the matrix at the given position

        Parameters
        ----------
        row_number : int
            number of row where the entry is located
        column_number : int
            number of the columne where the entry is located

        Returns
        -------
        float
            value given at the desired position

        Examples
        --------
        >>> from m_n_kappa.matrices import Matrix
        >>> a_matrix = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> a_matrix.entry(1, 1)
        4.0
        """
        return self.matrix[row_number][column_number]

    def multiply_by(self, multiplicant):
        """
        multiplies this matrix with the given multiplicant

        assumes that this matrix is on the left and the multiplicant is on
        the right of the multiplication-sign

        Parameters
        ----------
        multiplicant : Matrix | Vector | float
            Multiplicant to multiply this matrix with

        Returns
        -------
        Matrix | Vector
            depending on the input the multiplication results in Matrix or a Vector

        Raises
        ------
        ValueError
             in case multiplicant is either of type :py:class:`Matrix`,
             nor :py:class:`Matrix` nor float

        Examples
        --------
        >>> from m_n_kappa.matrices import Matrix
        >>> this_matrix = Matrix([[3, 2, 1], [1, 0, 2]])

        Multiply another Matrix:

        >>> other_matrix = Matrix([[1, 2], [0, 1], [4, 0]])
        >>> this_matrix.multiply_by(other_matrix)
        Matrix([[7.0, 8.0], [9.0, 2.0]])

        Multiply a vector:

        >>> a_vector = Vector([1, 2, 3])
        >>> this_matrix.multiply_by(a_vector)
        Vector([1.0, 26.0])

        Multiply a number (=scalar):

        >>> this_matrix.multiply_by(5)
        Matrix([[5.0, -15.0, 10.0], [5.0, 10.0, 35.0]])
        """
        if isinstance(multiplicant, Matrix):
            return self._matrix_multiplication(multiplicant)
        elif isinstance(multiplicant, Vector):
            return self._vector_multiplication(multiplicant)
        elif isinstance(multiplicant, (float, int)):
            return self._scalar_multiplication(float(multiplicant))
        else:
            raise ValueError(
                f"'multiplicant' must be of type 'Matrix', 'Vector' or 'float'."
                f"It is of type {type(multiplicant)}"
            )

    def add(self, matrix):
        """
        add this matrix to another matrix

        Parameters
        ----------
        matrix : Matrix
            the Matrix this matrix is added to

        Returns
        -------
        Matrix
            new Matrix where each entry is added to the corresponding one in the other matrix

        Raises
        ------
        TypeError
            if ``matrix`` is not of type :py:class:`m_n_kappa.matrices.Matrix`

        Examples
        --------
        >>> from m_n_kappa.matrices import Matrix
        >>> this_matrix = Matrix([[1, -3, 2], [1, 2, 7]])
        >>> other_matrix = Matrix([[0, 3, 5], [2, 1, -1]])
        >>> this_matrix.add(other_matrix)
        Matrix([[1.0, 0.0, 7.0], [3.0, 3.0, 6.0]])
        """
        if not isinstance(matrix, Matrix):
            raise TypeError(
                f"'matrix' must be of type 'Matrix'. Is of type '{type(matrix)}'"
            )

        if (
            self.column_number != matrix.column_number
            or self.row_number != matrix.row_number
        ):
            raise ValueError(
                f"This matrix must have the same number of rows (={self.row_number}) and the "
                f"columns (={self.column_number}) like the other one ({matrix.row_number}, "
                f"{matrix.column_number})."
            )
        new_matrix = []
        for row_number, row in enumerate(self.matrix):
            new_row = [
                entry + matrix.entry(row_number, column_number)
                for column_number, entry in enumerate(row)
            ]
            new_matrix.append(new_row)
        return Matrix(new_matrix)

    def subtract(self, matrix):
        """
        subtracts the given matrix form this matrix

        Parameters
        ----------
        matrix : Matrix
            matrix that is subtracted from this one

        Returns
        -------
        Matrix
            New Matrix where each value of the given matrix is subtracted from this one

        Examples
        --------
        >>> from m_n_kappa.matrices import Matrix
        >>> this_matrix = Matrix([[1, -3, 2], [1, 2, 7]])
        >>> other_matrix = Matrix([[0, 3, 5], [2, 1, -1]])
        >>> this_matrix.subtract(other_matrix)
        Matrix([[1.0, -6.0, -3.0], [-1.0, 1.0, 8.0]])
        """
        return self.add(matrix.multiply_by(-1.0))

    def row_vector(self, row_number: int) -> Vector:
        """
        vector of the given row in the matrix

        Parameters
        ----------
        row_number : int
            number of the row

        Returns
        -------
        Vector
           vector of the row given by the number
        """
        return Vector(self.row(row_number))

    def column_vector(self, column_number: int) -> Vector:
        """
        vector of the given column in the matrix

        Parameters
        ----------
        column_number : int
            number of the column

        Returns
        -------
        Vector
            vector of the column given by the number
        """
        return Vector(self.column(column_number))

    def _matrix_multiplication(self, right_matrix):
        """
        multiplicates two matrices on each other

        Parameters
        ----------
        right_matrix : Matrix
            matrix on the right side to multiply with this one

        Returns
        -------
        Matrix
            resulting Matrix from the matrix-multiplication
        """
        if self.column_number != right_matrix.row_number:
            raise ValueError(
                f"The column-number of this matrix (={self.column_number}) must be equal "
                f"to the row-number of the right-matrix (={right_matrix.row_number})"
            )
        matrix = []
        for row_number in range(len(self.matrix)):
            matrix.append([])
            for column_number in range(len(right_matrix.matrix[0])):
                scalar = self.row_vector(row_number).scalar_product(
                    right_matrix.column_vector(column_number)
                )
                matrix[row_number].append(scalar)
        return Matrix(matrix)

    def _vector_multiplication(self, vector: Vector) -> Vector:
        """
        multiply this matrix with a vector

        Parameters
        ----------
        vector : Vector

        Returns
        -------
        Vector
            vector resulting from the multiplication

        Raises
        ------
        ValueError
            in case this matrix has not the same number of columns as the vector has entries
        """
        if len(vector) != self.column_number:
            raise ValueError(
                f"This matrix must have the same number of columns (={self.column_number})"
                f" as the vector has entries (={len(vector)})."
            )
        new_vector = [
            self.row_vector(index).scalar_product(vector)
            for index in range(len(self.matrix))
        ]
        return Vector(new_vector)

    def _scalar_multiplication(self, scalar: float):
        """
        multiply a scalar with this matrix

        Parameters
        ----------
        scalar : float
            scalar to multiply this matrix with

        Returns
        -------
        Matrix
            This matrix where each entry is multiplied with ``scalar``
        """
        new_matrix = []
        for row in self.matrix:
            new_matrix.append([scalar * entry for entry in row])
        return Matrix(new_matrix)

    def orthonormal_triangular(self, algorithm="Givens-rotation"):
        """
        Determine orthogonal Matrix :math:`Q` and (upper) triangular Matrix :math:`R` of this matrix

        Parameters
        ----------
        algorithm : str
           Algorithm for computation of :math:`Q` and :math:`R`.
           Currently following algorithms are supported:

           - ``'Gram-Schmidt'`` :  `Gram-Schmidt <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`_
           - ``'Modified Gram-Schmidt'`` : Modified Gram-Schmidt
           - ``'Givens-rotation'``: `Givens rotation <https://en.wikipedia.org/wiki/Givens_rotation>`_

        Returns
        -------
        tuple[Matrix, Matrix]
           First entry is the orthogonal matrix of this matrix.
           Second entry is the (upper) triangluar Matrix of this matrix.
        """
        if algorithm.upper() == "GRAM-SCHMIDT":
            return self._gram_schmidt()
        elif algorithm.upper() == "MODIFIED GRAM-SCHMIDT":
            return self._gram_schmidt_modified()
        elif algorithm.upper() == "GIVENS-ROTATION":
            return self._givens()
        else:
            raise ValueError(
                f"Algorithm {algorithm} is not available."
                f"Possible algorithms are 'Gram-Schmidt' and 'Modified Gram-Schmidt'"
            )

    def replace(self, row: int, column: int, value: float) -> None:
        self._matrix[row][column] = value

    def _gram_schmidt(self) -> tuple:
        """
        Gram-Schmidt process to determine orthogonal and triangular matrix

        Returns
        -------
        tuple[Matrix, Matrix]
            Orthogonal Matrix :math:`Q` and triangular Matrix :math:`R`

        See Also
        --------
        :ref:`Gram-Schmidt on Wikipedia <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`
        """
        r = Matrix([])  # Triangular Matrix
        q = Matrix([])  # Orthogonal Matrix
        for column in range(self.column_number):
            r_j = Vector([0.0] * self.row_number)
            q_j = self.column_vector(column)
            for row in range(column):
                r_ij = q.column_vector(row).scalar_product(self.column_vector(column))
                r_j.replace(number=row, value=r_ij, inplace=True)
                q_j = q_j - q.column_vector(row).multiply_scalar(r_ij)
            r_j.replace(column, q_j.norm(), inplace=True)
            q_j = q_j.multiply_scalar(1.0 / r_j.entries[column])
            q = q.append(q_j)
            r = r.append(r_j)
        return q, r

    def _gram_schmidt_modified(self):
        """
        Gram-Schmidt process to determine orthogonal and triangular matrix

        Compared to original Gram-Schmidt this process is numerically more stable.

        Returns
        -------
        tuple[Matrix, Matrix]
            Orthogonal Matrix :math:`Q` and triangular Matrix :math:`R`
        """
        r = Matrix([])  # Triangular Matrix
        q = Matrix([])  # Orthogonal Matrix
        for column in range(self.column_number):
            r_j = Vector([0.0] * self.row_number)
            q_j = self.column_vector(column)
            for row in range(column):
                r_ij = q.column_vector(row).scalar_product(
                    q_j
                )  # <-- difference to Gram-Schmidt
                r_j.replace(number=row, value=r_ij, inplace=True)
                q_j = q_j - q.column_vector(row).multiply_scalar(r_ij)
            r_j.replace(column, q_j.norm(), inplace=True)
            if r_j.entries[column] == 0.0:
                q_j = q_j.multiply_scalar(1.0 / r_j.entries[column])
            q = q.append(q_j)
            r = r.append(r_j)
        return q, r

    def _givens(self):
        """
        QR-decomposition utilizing Givens rotation.

        Returns
        -------
        tuple[Matrix, Matrix]
            Orthogonal Matrix :math:`Q` and triangular Matrix :math:`R`
        """
        r = Matrix(self.matrix)
        rotation_matrices = []
        q = None
        for column in range(self.column_number):
            for row in range(self.row_number):
                if column >= row:
                    continue
                if r.entry(row, column) == 0.0:
                    continue
                diagonal_value = r.entry(column, column)
                position_value = r.entry(row, column)
                rho = (diagonal_value**2.0 + position_value**2.0) ** 0.5
                rotation_matrix = Identity(self.row_number)
                diagonal_value = diagonal_value / rho
                position_value = position_value / rho

                rotation_matrix.replace(row, row, diagonal_value)
                rotation_matrix.replace(column, column, diagonal_value)
                rotation_matrix.replace(row, column, (-1.0) * position_value)
                rotation_matrix.replace(column, row, position_value)
                rotation_matrices.append(rotation_matrix)

                r = rotation_matrix.multiply_by(r)

                if q is None:
                    q = rotation_matrix
                else:
                    q = rotation_matrix.multiply_by(q)
        if q is None:
            q = r
        return q.transpose(), r

    def diagonal(self):
        """
        build new matrix inheriting the diagonal from this matrix
        and put all other places to zero
        """
        matrix = Matrix([[0] * self.row_number] * self.row_number)
        for index in range(self.row_number):
            matrix.replace(
                row=index,
                column=index,
                value=self.entry(row_number=index, column_number=index),
            )
        return matrix

    def switch_rows(self, row_number_1: int, row_number_2: int, inplace: bool = True):
        """
        Switch the rows with the given numbers

        Parameters
        ----------
        row_number_1 : int
            position of the row to be switched
        row_number_2 : int
            position the row is to be switched to
        inplace : bool
            ``False`` gives a new Matrix back, ``True`` makes the change in place

        Returns
        -------
        Matrix | None
        """
        if row_number_1 > row_number_2:
            row_number_1, row_number_2 = row_number_2, row_number_1
        if inplace:
            self._matrix[row_number_1], self._matrix[row_number_2] = (
                self._matrix[row_number_2],
                self._matrix[row_number_1],
            )
        else:
            matrix = self
            matrix.switch_rows(row_number_1, row_number_2)
            return matrix

    def lu_decomposition(self, use_pivot=True):
        """
        Decompose matrix in a lower/left and a upper/right matrix

        Returns
        -------
        tuple[Matrix, Matrix, Matrix]
            lower-matrix, upper-matrix and pivot-matrix
        """
        u_matrix = self
        l_matrix = Identity(self.row_number, diagonal_value=0.0)
        p_matrix = Identity(self.row_number)
        for column_number in range(self.column_number - 1):
            # Pivot
            if use_pivot:
                max_column_value = max(
                    u_matrix.column(column_number)[column_number:], key=abs
                )
                max_value_index = u_matrix.column(column_number).index(max_column_value)
                if max_value_index > column_number:
                    for matrix in [u_matrix, l_matrix, p_matrix]:
                        matrix.switch_rows(column_number, max_value_index)
            for row_number in range(column_number + 1, self.row_number):
                if u_matrix.entry(column_number, column_number) != 0.0:
                    multiply_value = u_matrix.entry(
                        row_number, column_number
                    ) / u_matrix.entry(column_number, column_number)
                    l_matrix.replace(row_number, column_number, multiply_value)
                else:
                    multiply_value = 0.0
                for rest_column in range(column_number, self.column_number):
                    replace_value = u_matrix.entry(
                        row_number, rest_column
                    ) - multiply_value * u_matrix.entry(column_number, rest_column)
                    u_matrix.replace(row_number, rest_column, replace_value)
        l_matrix = l_matrix + Identity(self.row_number)
        return l_matrix, u_matrix, p_matrix

    def determinant(self) -> float:
        """compute the determinant of the matrix using LU-decomposition"""
        _, u_matrix, _ = self.lu_decomposition()
        return (-1) * reduce(
            operator.mul, (row[index] for index, row in enumerate(u_matrix.matrix))
        )


class Identity(Matrix):

    """
    Identity Matrix

    .. versionadded:: 0.1.0

    Where the diagonal values have a given value
    and all other values are zero.
    """

    def __init__(self, row_column_number: int, diagonal_value: float = 1.0):
        """
        Parameters
        ----------
        row_column_number : int
            number of rows and columns of the matrix
        diagonal_value : float
            value along the diagonal (Default: 1.0)
        """
        super().__init__(matrix=[[0] * row_column_number] * row_column_number)
        self._add_diagonal(diagonal_value)

    def _add_diagonal(self, value=1.0):
        """add the diagonal to the given matrix"""
        for index in range(self.row_number):
            self.replace(row=index, column=index, value=value)


class LinearEquationsSystem:

    """
    Solve System of linear equations of type :math:`\\mathbf{A} \\vec{x} = \vec{b}`

    .. versionadded:: 0.2.0
    """

    __slots__ = "_coefficients", "_constants"

    def __init__(self, coefficients: Matrix, constants: Vector):
        """
        Parameters
        ----------
        coefficients : Matrix
           matrix of coefficients :math:`\\mathbf{A}`
        constants : Vector
           results-vector :math:`\\vec{b}`

        Examples
        --------
        The following Formula is solved easily by :py:class:`m_n_kappa.matrices.LinearEquationSystem`

        .. math::

           \\mathbf{A} \\vec{x} = \\vec{b}

           \\text{with}

           \\mathbf{A} = \\begin{bmatrix}
              4 & -2 & 1 \\\\
              -1 & 3 & 4 \\\\
              5 & -1 & 3
           \\end{bmatrix} \\text{ and } \vec{b} = \\begin{bmatrix} 15 \\\\ 15 \\\\ 26 \\begin{bmatrix}

        To define the matrices follow the following steps.

        >>> from m_n_kappa.matrices import Matrix, Vector, LinearEquationsSystem
        >>> coefficients = Matrix([[4, -2, 1], [-1, 3, 4], [5, -1, 3]])
        >>> constants = Vector([15, 15, 26])

        The above given Formula is then solved as follows.

        >>> LinearEquationsSystem(coefficients, constants).solve()
        Vector([2.0, -1.0, 5.0])

        """
        self._coefficients = coefficients
        self._constants = constants

    def __repr__(self) -> str:
        return (
            f"LinearEquationsSystem(\n"
            f"\tcoefficients={self.coefficients},\n"
            f"\tconstants={self.constants})"
        )

    @property
    def coefficients(self) -> Matrix:
        """coefficients-matrix"""
        return self._coefficients

    @property
    def constants(self) -> Vector:
        """constants, i.e. results of the equation"""
        return self._constants

    def solve(self, solver: str = "QR") -> Vector:
        """solve the system of linear equations using the"""
        if solver.upper() == "QR":
            return self.qr_decomposition()
        elif solver.upper() == "LU":
            return self.lu_decomposition()
        else:
            raise ValueError(
                f"Solver {solver} is not available. "
                f'Possible values are "QR" for "QR-Decomposition" and '
                f'"LU" for "LU-Decomposition"'
            )

    def qr_decomposition(self) -> Vector:
        """
        solver of system of linear equations using
        `QR-decomposition <https://en.wikipedia.org/wiki/QR_decomposition>`
        """
        q, r = self.coefficients.orthonormal_triangular()
        q_t: Matrix = q.transpose()
        z: Vector = q_t.multiply_by(self.constants)
        return self.substitution(triangle=r, constants=z, make_forward=False)

    def lu_decomposition(self) -> Vector:
        """
        solver of system of linear equations using
        `LU-decomposition <https://en.wikipedia.org/wiki/LU_decomposition>`
        """
        l_matrix, u_matrix, p_matrix = self.coefficients.lu_decomposition()
        b = p_matrix.multiply_by(self.constants)
        y = self.substitution(triangle=l_matrix, constants=b, make_forward=True)
        return self.substitution(triangle=u_matrix, constants=y, make_forward=False)

    @staticmethod
    def substitution(
        triangle: Matrix, constants: Vector, make_forward: bool = True
    ) -> Vector:
        """
        solving a system of linear equations consisting of a triangle-matrix
        and resulting constants by substitution

        Parameters
        ----------
        triangle : Matrix
            triangle matrix, where all entries below the diagonal denote to zero
        constants : Vector
           Resulting vector (given on the right hand side of a Formula)
        make_forward : bool

           - ``True``: make forward-substitution
           - ``False``: make backward-substitution (Default)

        Returns
        -------
        Vector
            Variables leading to the given result
        """

        last_row = len(constants)
        x = Vector([0.0] * last_row)  # Variables
        if make_forward:
            the_range = range(last_row)
        else:
            the_range = range(last_row - 1, -1, -1)
        for row in the_range:
            subtractor = triangle.row_vector(row).scalar_product(x)
            numerator = constants.entries[row] - subtractor
            denominator = triangle.entry(row, row)
            if numerator == 0.0 and denominator == 0.0:
                x_i = 0.0
            else:
                x_i = numerator / denominator
            x.replace(row, x_i, inplace=True)
        return x
