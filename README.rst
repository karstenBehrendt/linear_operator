
LinearOperator
==============


.. image:: https://github.com/cornellius-gp/linear_operator/actions/workflows/run_test_suite.yml/badge.svg
   :target: https://github.com/cornellius-gp/linear_operator/actions/workflows/run_test_suite.yml
   :alt: Test


.. image:: https://readthedocs.org/projects/linear-operator/badge/?version=latest
   :target: https://linear-operator.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation


.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License



.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version


.. image:: https://img.shields.io/conda/v/gpytorch/linear_operator.svg
   :target: https://anaconda.org/gpytorch/linear_operator
   :alt: Conda


.. image:: https://img.shields.io/pypi/v/linear_operator.svg
   :target: https://pypi.org/project/linear_operator
   :alt: PyPI


.. docs_index_start


LinearOperator is a PyTorch package for abstracting away the linear algebra routines needed for structured matrices (or operators).

**This package is in beta.**
Currently, most of the functionality only supports positive semi-definite and triangular matrices.
Package development TODOs:


* [x] Support PSD operators
* [x] Support triangular operators
* [ ] Interface to specify structure (i.e. symmetric, triangular, PSD, etc.)
* [ ] Add algebraic routines for symmetric operators
* [ ] Add algebraic routines for generic square operators
* [ ] Add algebraic routines for generic rectangular operators
* [ ] Add sparse operators

To get started, run either

.. code-block:: sh

   pip install linear_operator
   # or
   conda install linear_operator -c gpytorch

or `see below <#installation>`_ for more detailed instructions.

Why LinearOperator
------------------

Before describing what linear operators are and why they make a useful abstraction, it's easiest to see an example.
Let's say you wanted to compute a matrix solve:

.. math::
  \boldsymbol A^{-1} \boldsymbol b.

If you didn't know anything about the matrix :math:`\boldsymbol A`, the simplest (and best) way to accomplish this in code is:

.. code-block:: python

   # A = torch.randn(1000, 1000)
   # b = torch.randn(1000)
   torch.linalg.solve(A, b)  # computes A^{-1} b

While this is easy, the ``solve`` routine is :math:`\mathcal O(N^3)`, which gets very slow as :math:`N` grows large.

However, let's imagine that we knew that :math:`\boldsymbol A` was equal to a low rank matrix plus a diagonal
(i.e. :math:`\boldsymbol A = \boldsymbol C \boldsymbol C^\top + \boldsymbol D`
for some skinny matrix :math:`\boldsymbol C` and some diagonal matrix :math:`\boldsymbol D`.)
There's now a very efficient :math:`\boldsymbol O(N)` routine to compute :math:`\boldsymbol A^{-1}` (the `Woodbury formula <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_\ ).
**In general**, if we know that :math:`\boldsymbol A` has structure,
we want to use efficient linear algebra routines - rather than the general routines -
that exploit this structure.

Without LinearOperator
^^^^^^^^^^^^^^^^^^^^^^

Implementing the efficient solve that exploits :math:`\boldsymbol A`'s low-rank-plus-diagonal structure would look something like this:

.. code-block:: python

   def low_rank_plus_diagonal_solve(C, d, b):
       # A = C C^T + diag(d)
       # A^{-1} b = D^{-1} b - D^{-1} C (I + C^T D^{-1} C)^{-1} C^T D^{-1} b
       #   where D = diag(d)

       D_inv_b = b / d
       D_inv_C = C / d.unsqueeze(-1)
       eye = torch.eye(C.size(-2))
       return (
           D_inv_b - D_inv_C @ torch.cholesky_solve(
               C.mT @ D_inv_b,
               torch.linalg.cholesky(eye + C.mT @ D_inv_C, upper=False),
               upper=False
           )
       )


   # C = torch.randn(1000, 20)
   # d = torch.randn(1000)
   # b = torch.randn(1000)
   low_rank_plus_diagonal_solve(C, d, b)  # computes A^{-1} b in O(N) time, instead of O(N^3)

While this is efficient code, it's not ideal for a number of reasons:


#. It's a lot more complicated than ``torch.linalg.solve(A, b)``.
#. There's no object that represents :math:`\boldsymbol A`.
   To perform any math with :math:`\boldsymbol A`, we have to pass around the matrix ``C`` and the vector ``d``.

With LinearOperator
^^^^^^^^^^^^^^^^^^^

The LinearOperator package offers the best of both worlds:

.. code-block:: python

   from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator
   # C = torch.randn(1000, 20)
   # d = torch.randn(1000)
   # b = torch.randn(1000)
   A = LowRankRootLinearOperator(C) + DiagLinearOperator(d)  # represents C C^T + diag(d)

it provides an interface that lets us treat :math:`\boldsymbol A` as if it were a generic tensor,
using the standard PyTorch API:

.. code-block:: python

   torch.linalg.solve(A, b)  # computes A^{-1} b efficiently!

Under-the-hood, the ``LinearOperator`` object keeps track of the algebraic structure of :math:`\boldsymbol A` (low rank plus diagonal)
and determines the most efficient routine to use (the Woodbury formula).
This way, we can get a efficient :math:`\mathcal O(N)` solve while abstracting away all of the details.

Crucially, :math:`\boldsymbol A` is never explicitly instantiated as a matrix, which makes it possible to scale
to very large operators without running out of memory:

.. code-block:: python

   # C = torch.randn(10000000, 20)
   # d = torch.randn(10000000)
   # b = torch.randn(10000000)
   A = LowRankRootLinearOperator(C) + DiagLinearOperator(d)  # represents a 10M x 10M matrix!
   torch.linalg.solve(A, b)  # computes A^{-1} b efficiently!


.. docs_index_end
.. docs_abstraction_start



What is a Linear Operator?
--------------------------

A linear operator is a generalization of a matrix.
It is a linear function that is defined in by its application to a vector.
The most common linear operators are (potentially structured) matrices,
where the function applying them to a vector are (potentially efficient)
matrix-vector multiplication routines.

In code, a ``LinearOperator`` is a class that


#. specifies the tensor(s) needed to define the LinearOperator,
#. specifies a ``_matmul`` function (how the LinearOperator is applied to a vector),
#. specifies a ``_size`` function (how big is the LinearOperator if it is represented as a matrix, or batch of matrices), and
#. specifies a ``_transpose_nonbatch`` function (the adjoint of the LinearOperator).
#. (optionally) defines other functions (e.g. ``logdet``\ , ``eigh``\ , etc.) to accelerate computations for which efficient sturcture-exploiting routines exist.

For example:

.. code-block:: python

   class DiagLinearOperator(linear_operator.LinearOperator):
       r"""
       A LinearOperator representing a diagonal matrix.
       """
       def __init__(self, diag):
           # diag: the vector that defines the diagonal of the matrix
           self.diag = diag

       def _matmul(self, v):
           return self.diag.unsqueeze(-1) * v

       def _size(self):
           return torch.Size([*self.diag.shape, self.diag.size(-1)])

       def _transpose_nonbatch(self):
           return self  # Diagonal matrices are symmetric

       # this function is optional, but it will accelerate computation
       def logdet(self):
           return self.diag.log().sum(dim=-1)
   # ...

   D = DiagLinearOperator(torch.tensor([1., 2., 3.])
   # Represents the matrix
   #   [[1., 0., 0.],
   #    [0., 2., 0.],
   #    [0., 0., 3.]]
   torch.matmul(D, torch.tensor([4., 5., 6.])
   # Returns [4., 10., 18.]

While ``_matmul``\ , ``_size``\ , and ``_transpose_nonbatch`` might seem like a limited set of functions,
it turns out that most functions on the ``torch`` and ``torch.linalg`` namespaces can be efficiently implemented
using only these three primitative functions.

Moreover, because ``_matmul`` is a linear function, it is very easy to compose linear operators in various ways.
For example: adding two linear operators (\ ``SumLinearOperator``\ ) just requires adding the output of their ``_matmul`` functions.
This makes it possible to define very complex compositional structures that still yield efficient linear algebraic routines.

Finally, ``LinearOperator`` objects can be composed with one another, yielding new ``LinearOperator`` objects and automatically keeping track of algebraic structure after each computation.
As a result, users never need to reason about what efficient linear algebra routines to use  (so long as the input elements defined by the user encode known input structure).
See the `using LinearOperator objects <#using-linearoperator-objects>`_ section for more details.

Use Cases
---------

There are several use cases for the LinearOperator package.
Here we highlight two general themes:

Modular Code for Structured Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example, let's say that you have a generative model that involves
sampling from a high-dimensional multivariate Gaussian.
This sampling operation will require storing and manipulating a large covariance matrix,
so to speed things up you might want to experiment with different structured
approximations of that covariance matrix.
This is easy with the LinearOperator package.

.. code-block:: python

   from gpytorch.distributions import MultivariateNormal

   # variance = torch.randn(10000)
   cov = DiagLinearOperator(variance)
   # or
   # cov = LowRankRootLinearOperator(...) + DiagLinearOperator(...)
   # or
   # cov = KroneckerProductLinearOperator(...)
   # or
   # cov = ToeplitzLinearOperator(...)
   # or
   # ...

   mvn = MultivariateNormal(torch.zeros(cov.size(-1), cov) # 10000-dimensional MVN
   mvn.rsample()  # returns a 10000-dimensional vector

Efficient Routines for Complex Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many of the efficient linear algebra routines in LinearOperator are iterative algorithms
based on matrix-vector multiplication.
Since matrix-vector multiplication obeys many nice compositional properties
it is possible to obtain efficient routines for extremely complex compositional LienarOperators:

.. code-block:: python

   from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator, ToeplitzLinearOperator

   # mat1 = 200 x 200 PSD matrix
   # mat2 = 100 x 100 PSD matrix
   # vec3 = 20000 vector

   A = KroneckerProductLinearOperator(mat1, mat2) + RootLinearOperator(ToeplitzLinearOperator(vec3))
   # represents a 20000 x 20000 matrix

   torch.linalg.solve(A, torch.randn(20000))  # Sub O(N^3) routine!


.. docs_abstraction_end


Using LinearOperator Objects
----------------------------

LinearOperator objects share (mostly) the same API as ``torch.Tensor`` objects.
Under the hood, these objects use ``__torch_function__`` to dispatch all efficient linear algebra operations
to the ``torch`` and ``torch.linalg`` namespaces.
This includes


* ``torch.add``
* ``torch.cat``
* ``torch.clone``
* ``torch.diagonal``
* ``torch.dim``
* ``torch.div``
* ``torch.expand``
* ``torch.logdet``
* ``torch.matmul``
* ``torch.numel``
* ``torch.permute``
* ``torch.prod``
* ``torch.squeeze``
* ``torch.sub``
* ``torch.sum``
* ``torch.transpose``
* ``torch.unsqueeze``
* ``torch.linalg.cholesky``
* ``torch.linalg.eigh``
* ``torch.linalg.eigvalsh``
* ``torch.linalg.solve``
* ``torch.linalg.svd``

Each of these functions will either return a ``torch.Tensor``\ , or a new ``LinearOperator`` object,
depending on the function.
For example:

.. code-block:: python

   # A = RootLinearOperator(...)
   # B = ToeplitzLinearOperator(...)
   # d = vec

   C = torch.matmul(A, B)  # A new LienearOperator representing the product of A and B
   torch.linalg.solve(C, d)  # A torch.Tensor

For more examples, see the `examples folder <https://github.com/cornellius-gp/linear_operator/blob/main/examples/>`_.

Batch Support and Broadcasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LinearOperator`` objects operate naturally in batch mode.
For example, to represent a batch of 3 ``100 x 100`` diagonal matrices:

.. code-block:: python

   # d = torch.randn(3, 100)
   D = DiagLinearOperator(d)  # Reprents an operator of size 3 x 100 x 100

These objects fully support broadcasted operations:

.. code-block:: python

   D @ torch.randn(100, 2)  # Returns a tensor of size 3 x 100 x 2

   D2 = DiagLinearOperator(torch.randn([2, 1, 100]))  # Represents an operator of size 2 x 1 x 100 x 100
   D2 + D  # Represents an operator of size 2 x 3 x 100 x 100

Indexing
^^^^^^^^

``LinearOperator`` objects can be indexed in ways similar to torch Tensors. This includes:


* Integer indexing (get a row, column, or batch)
* Slice indexing (get a subset of rows, columns, or batches)
* LongTensor indexing (get a set of individual entries by index)
* Ellipses (support indexing operations with arbitrary batch dimensions)

.. code-block:: python

   D = DiagLinearOperator(torch.randn(2, 3, 100))  # Represents an operator of size 2 x 3 x 100 x 100
   D[-1]  # Returns a 3 x 100 x 100 operator
   D[..., :10, -5:]  # Returns a 2 x 3 x 10 x 5 operator
   D[..., torch.LongTensor([0, 1, 2, 3]), torch.LongTensor([0, 1, 2, 3])]  # Returns a 2 x 3 x 4 tensor

Composition and Decoration
^^^^^^^^^^^^^^^^^^^^^^^^^^

LinearOperators can be composed with one another in various ways.
This includes


* Addition (\ ``LinearOpA + LinearOpB``\ )
* Matrix multiplication (\ ``LinearOpA @ LinearOpB``\ )
* Concatenation (\ ``torch.cat([LinearOpA, LinearOpB], dim=-2)``\ )
* Kronecker product (\ ``torch.kron(LinearOpA, LinearOpB)``\ )

In addition, there are many ways to "decorate" LinearOperator objects.
This includes:


* Elementwise multiplying by constants (\ ``torch.mul(2., LinearOpA)``\ )
* Summing over batches (\ ``torch.sum(LinearOpA, dim=-3)``\ )
* Elementwise multiplying over batches (\ ``torch.prod(LinearOpA, dim=-3)``\ )

See the documentation for a `full list of supported composition and decoration operations <https://linear-operator.readthedocs.io/en/latest/composition_decoration_operators.html>`_.

Installation
------------

LinearOperator requires Python >= 3.8.

Standard Installation (Most Recent Stable Version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend installing via ``pip`` or Anaconda:

.. code-block:: sh

   pip install linear_operator
   # or
   conda install linear_operator -c gpytorch

The installation requires the following packages:


* PyTorch >= 1.11
* Scipy

You can customize your PyTorch installation (i.e. CUDA version, CPU only option)
by following the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

Installing from the ``main`` Branch (Latest Unsable Version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install what is currently on the ``main`` branch (potentially buggy and unstable):

.. code-block:: sh

   pip install --upgrade git+https://github.com/cornellius-gp/linear_operator.git

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

If you are contributing a pull request, it is best to perform a manual installation:

.. code-block:: sh

   git clone https://github.com/cornellius-gp/linear_operator.git
   cd linear_operator
   pip install -e .[dev,test]

To generate the documentation locally, you will also need to run the following command
from the linear_operator folder:

.. code-block:: sh

   pip install -r docs/requirements.txt

Contributing
------------

See the contributing guidelines `CONTRIBUTING.md <https://github.com/cornellius-gp/linear_operator/blob/main/CONTRIBUTING.md>`_
for information on submitting issues and pull requests.

License
-------

LinearOperator is `MIT licensed <https://github.com/cornellius-gp/linear_operator/blob/main/LICENSE>`_.
