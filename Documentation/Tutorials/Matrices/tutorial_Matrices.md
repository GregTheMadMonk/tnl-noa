\page tutorial_Matrices  Matrices tutorial

## Introduction

TNL offers the following type of matrices:  dense matrices, sparse matrices, tridiagonal matrices, multidiagonal matrices and lambda matrices. The sparse matrices can be marked as symmetric to optimize memory requirements. The interfaces of given matrix types are designed to be as unified as possible to ensure that the user can easily switch between different matrix types while making no or only a little changes in the source code. All matrix types allows traversing all matrix elements and manipulate them using a lambda function as well as performing flexible reduction in matrix rows. The following text describes particular matrix types in details.


## Table of Contents
1. [Dense matrices](#dense_matrices)
2. [Sparse matrices](#sparse_matrices)
3. [Tridiagonal matrices](#tridiagonal_matrices)
4. [Multidiagonal matrices](#multidiagonal_matrices)
5. [Lambda matrices](#lambda_matrices)


## Dense matrices <a name="dense_matrices"></a>

Dense matrix is a templated class defined in namespace \ref TNL::Matrices. It has five template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for GPU supporting CUDA. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default it is the row-major order if the matrix is allocated in the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.

### Dense matrix allocation and initiation

The following examples show how to allocate the dense matrix and how to initialize the matrix elements. Small matrices can be created simply by the constructor with an initializer list.

\include Matrices/DenseMatrix/DenseMatrixExample_Constructor_init_list.cpp

In fact, the constructor takes a list of initializer lists. Each embedded list defines one matrix row and so the number of matrix rows is given by the size of the outer initializer list.  The number of matrix columns is given by the longest inner initializer lists. Shorter inner lists are filled with zeros from the right side. The result looks as follows:

\include DenseMatrixExample_Constructor_init_list.out

Larger matrices can be set-up with methods `setElement` and `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement). The following example shows how to call these methods from the host.

\include DenseMatrixExample_addElement.cpp

As we can see, both methods can be called from the host no matter where the matrix is allocated. If it is on GPU, each call of `setElement` or `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement) causes slow transfer of tha data between CPU and GPU. Use this approach only if the performance is not a priority for example for matrices which are set only once this way. The result looks as follows:

\include DenseMatrixExample_addElement.out

More efficient way of the matrix initialization on GPU consists in calling the methods `setElement` and `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement) directly from GPU. It is demonstrated in the following example (of course it works even for CPU):

\include DenseMatrixExample_setElement.cpp

Here we use `SharedPointer` (\ref TNL::Pointers::SharedPointer) to make the matrix accessible in lambda functions even on GPU. We first call the `setElement` method from CPU to set the `i`-th diagonal element to `i`. Next we iterate over the matrix rows with `ParallelFor`and for each row we call a lambda function `f`. This is done on the same device where the matrix is allocated and so it is more efficient for matrices allocated on GPU. In the lambda function we just set the `i`-th diagonal element to `-i`. The result looks as follows:

\include DenseMatrixExample_setElement.out

If we want to set more matrix elements in each row, we can use inner for-loop in the lambda function `f`. This, however, is limiting the parallelization and it can be inefficient for larger matrices. The next example demonstrates a method `forRows` (\ref TNL::Matrices::DenseMatrix::forRows) which iterates over all matrix elements in parallel and it calls a lambda function defining an operation we want to do on the matrix elements.

\include DenseMatrixExample_forRows.cpp

Firstly note, that this is simpler since we do not need any `SharedPointer`. The lambda function `f` requires the following parameters:

* `rowIdx` is the row index of given matrix element.
* `columnIdx` is the column index of given matrix element.
* `value` is a reference on the matrix element value and so by changing this value we can modify the matrix element.
* `compute` is a boolean which, when set to `false`, indicates that we can skip the rest of the matrix row. This is, however, only a hint and it does not guarantee that the rest of the matrix row is really skipped.

The result looks as follows:

\include DenseMatrixExample_forRows.out

### Flexible reduction in matrix rows

Simillar operation to `forRows` is `rowsReduction` (\ref TNL::Matrices::DenseMatrix::rowsReduction) which performs given reduction in each matric row. For example, a matrix-vector product can be seen as a reduction of products of matrix elements and input vector in particular matrix rows. The first element of the result vector ios obtained as:

\f[
y_1 = a_{11} x_1 + a_{12} x_2 + \ldots + a_{1n} x_n = \sum_{j=1}^n a_{1j}x_j
\f]

and in general i-th element of the result vector is computed as

\f[
y_i = a_{i1} x_1 + a_{i2} x_2 + \ldots + a_{in} x_n = \sum_{j=1}^n a_{ij}x_j.
\f]

We see that in i-th matrix row we have to compute the sum \f$\sum_{j=1}^n a_{ij}x_j\f$ which is reduction of products \f$ a_{ij}x_j\f$. Similar to *flexible parallel reduction* (\ref TNL::Algorithms::Reduction) we just need to design proper lambda functions. See the following example:


\include DenseMatrixExample_rowsReduction_vectorProduct.cpp

The `fetch` lambda function computes the product \f$ a_{ij}x_j\f$ where \f$ a_{ij} \f$ is represented by `value` and \f$x_j \f$ is represented by `xView[columnIdx]`. The reduction is just sum of results particular products and it is represented by by the lambda function `reduce`. Finaly, the lambda function `keep` is responsible for storing the results of reduction in each matrix row (which is represented by the variable `value`) into the output vector `y`.  
The result looks as:

\include DenseMatrixExample_rowsReduction_vectorProduct.out

We will show one more example which is computation of maximal absolute value in each matrix row. The results will be stored in a vector:

\f[
y_i = \max_{j=1,\ldots,n} |a_{ij}|.
\f]

See the following example:

\include DenseMatrixExample_rowsReduction_maxNorm.cpp


The `fetch` lambda function just returns absolute value of \f$a_{ij} \f$ which is represented again by the varibale `value`. The `reduce` lambda function returns larger of given values and the lambda fuction 'keep' stores the results to the output vectro the same way as in the previous example. Of course, if we compute the maximum of all output vector elements we get some kined of max matrix norm. The output looks as:

\include DenseMatrixExample_rowsReduction_maxNorm.out

### Dense-matrix vector product

One of the most important matrix operation is the matrix-vector multiplication. It is represented by a method `vectorProduct` (\ref TNL::Matrices::DenseMatrix::vectorProduct). It is templated method with two template parameters `InVector` and `OutVector` telling the types of input and output vector respectively. Usually one will substitute some of \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView for these types. The method accepts the following parameters:

* `inVector` is the input vector having the same number of elements as the number of matrix columns.
* `outVector` is the output vector having the same number of elements as the number of matrix rows.
* `matrixMultiplicator` is a number by which the result of matrix-vector product is multiplied. 
* `outVectorMultiplicator` is a number by which the output vector is multiplied before added to the result of matrix-vector product.
* `begin` is an index of the first matrix row that is involved in the multiplication. It is zero be default.
* `end` is an index of the last matrix row that is involved in the multiplication. It is the last matrix row by default.

Note that the ouput vector dimension must be the same as the number of matrix rows no matter how we set `begin` and `end` parameters. These parameters just say that some matrix rows and the output vector elements are omitted.

To summarize, this method computes the following formula:

`outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector.`

### Dense matrix IO

The dense matrix can be saved to a file using a method `save` (\ref TNL::Matrices::DenseMatrix::save) and restored with a method `load` (\ref TNL::Matrices::DenseMatrix::load). To print the matrix a method `print` (\ref TNL::Matrices::DenseMatrix::print) can be used.

### Dense matrix view

Similar to array view (\ref TNL::Containers::ArayView) and vector view (\ref TNL::Containers::VectorView), matrices also offer their view for easier use with lambda functions. For the dense matrix there is a `DenseMatrixView` (\ref TNL::Matrcioes::DenseMatrixView).

TODO: Template parameters description

We will demonstrate it on the example showing the method `setElement` (\ref TNL::Matrices::DenseMatrix::setElement). However, the `SharedPointer` will be replaced with the `DenseMatrixView`. The code looks as follows:

\include DenseMatrixViewExample_setElement.cpp

And the result is:

\include DenseMatrixViewExample_setElement.out

The dense matrix view offers almost all methods which the dense matrix does. So it can be easily used at almost any situation the same way as the dense matrix itself.


TODO: Using DenseMatrixView for data encapsulation

## Sparse matrices <a name="sparse_matrices"></a>

[Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) arte extremely important in a lot of numerical algorithms. They are used at situations when we need to operate with matrices having majority of the matrix elements equal to zero. In this case, only the non-zero matrix elements are stored with possible some *padding zeros* used for memory alignment. This is necessary mainly on GPUs. Consider just matrix having 50,000 rows and columns whih is 2,500,000,000 matrix elements. If we store each matrix element in double precision (it means eight bytes per element) we need 20,000,000,000 bytes which is nearly 20 GB of memory. If there are only five non-zero elements in each row we need only \f$8 \times 5 \times 50,000=2,000,000\f$ bytes and so nearly 200 MB. It is really great difference.

Major disadventage of sparse matrices is that there are a lot of different formats for storing such matrices. Though [CSR - Compressed Sparse Row](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format is the most popular of all, especially for GPUs there are many other formats which perform differently on various matrices. So it is a good idea to test several sparse matrix formats if you want to get the best performance. In TNL, there is one templated class \ref TNL::Matrices::SparseMatrix representing the sparse matrices. The change of underlying matrix format can be done just by changing one template parameter. The list of the template paramaters is as follows:


* `Real` is type if the matrix elements. It is `double` by default.
* `Device` is a device where the matrix is allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for GPU supporting CUDA. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `MatrixType` tells if the matrix is symmetric (\ref TNL::Matrices::SymmetricMatrix) or general (\ref TNL::Matrices::GeneralMatrix). It is a \ref TNL::Matrices::GeneralMatrix by default.
* `Segments` define the format of the sparse matrix. It can be (by default, it is \ref TNL::Algorithms::Segments::CSR):
   * \ref TNL::Algorithms::Segments::CSR for [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)).
   * \ref TNL::Algorithms::Segments::Ellpack for [Ellpack format](http://mgarland.org/files/papers/nvr-2008-004.pdf).
   * \ref TNL::Algorithms::Segments::SlicedEllpack for [SlicedEllpack format](https://link.springer.com/chapter/10.1007/978-3-642-11515-8_10) which was also presented as [Row-grouped CSR format](https://arxiv.org/abs/1012.2270).
   * \ref TNL::Algorithms::Segments::ChunkedEllpack for [ChunkedEllpack format](http://geraldine.fjfi.cvut.cz/~oberhuber/data/vyzkum/publikace/12-heller-oberhuber-improved-rgcsr-format.pdf) which we reffered as Improved Row-grouped CSR and we renamed it to Ellpack format since it uses padding zeros.
   * \ref TNL::Algorithms::Segments::BiEllpack for [BiEllpack format](https://www.sciencedirect.com/science/article/pii/S0743731514000458?casa_token=2phrEj0Ef1gAAAAA:Lgf6rMBUN6T7TJne6mAgI_CSUJ-jR8jz7Eghdv6L0SJeGm4jfso-x6Wh8zgERk3Si7nFtTAJngg).
* `ComputeReal` is type which is used for internal computations. By default it is the same as `Real` if `Real` is not `bool`. If `Real` is `bool`, `ComputeReal` is set to `Index` type. This can be changed, of course, by the user.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type – see TNL::Allocators::Default.
* `IndexAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the column indexes of the matrix elements. By default, it is the default allocator for given `Index` type and `Device` type – see \ref TNL::Allocators::Default.

If `Real` is set to `bool`, we get *a binary matrix* for which the non-zero elements can be equal only to one and so the matrix elements values are not stored explicitly in the memory.

### Sparse matrix allocation and initiation

Small matrices can be initialized by a constructor with initializer list. We assume having the follwong sparse matrix

\f[
\left(
\begin{array}{ccccc}
 1 &  0 &  0 &  0 &  0 \\
-1 &  2 & -1 &  0 &  0 \\
 0 & -1 &  2 & -1 &  0 \\
 0 &  0 & -1 &  2 & -1 \\
 0 &  0 &  0 & -1 &  0
\end{array}
\right).
\f]

The following example shows how to create it using the initializer list constructor:

\include SparseMatrixExample_Constructor_init_list_2.cpp

The constructor accepts the following parameters:

* `rows` is a number of matrix rows.
* `columns` is a number of matrix columns.
* `data` is definition of non-zero matrix elements. It is a initializer list of triples having a form `{ row_index, column_index, value }`. In fact, it is very much like the coordinates format - [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)).

The constructor also accepts `Real` and `Index` allocators (\ref TNL::Allocators) but their are not important for this example. The result looks as follows:

\include SparseMatrixExample_Constructor_init_list_2.out

Larger matrices are created in two steps:

1. We use a method \ref TNL::Matrices::SparseMatrix::setRowCapacities to initialize the underlying matrix format and to allocate memory for the matrix elements. This method only needs to know how many non-zero elements are supposed to be in each row. Once this is set, it cannot be changed only by reseting the whole matrix. In most situations, this is not an issue to compute the number of non-zero elements in each row. Note, however, that we do not tell the positions of the non-zeto elements. If some matrix format needs this information it cannot be used with this implementation of the sparse matrix.
2. The non-zero matrix elements can be set-up. We insert one non-zero element after another by telling its coordinates and a value. Since probably all sparse matrix formats are designed to allow quick acces to particular matrix rows, this insertion is usualy quite efficient and can by done in parallel by mapping different threads to different matrix rows.

See the following example which creates lower triangular matrix like this one

\f[
\left(
\begin{array}{ccccc}
 1 &  0 &  0 &  0 &  0 \\
 2 &  1 &  0 &  0 &  0 \\
 3 &  2 &  1 &  0 &  0 \\
 4 &  3 &  2 &  1 &  0 \\
 5 &  4 &  3 &  2 &  1
\end{array}
\right).
\f]


\include SparseMatrixExample_setRowCapacities.cpp

The method \ref TNL::Matrices::SparseMatrix::setRowCapacities reads the required capacities of the matrix rows from a vector (or simmilar container - \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector and \ref TNL::Containers::VectorView) which has the same number of elements as the number of matrix rows and each element defines the capacity of the related row. The result looks as follows:

\include SparseMatrixExample_setRowCapacities.out

There are constructors which also set the row capacities, one uses a vector ...

\include SparseMatrixExample_Constructor_rowCapacities_vector.cpp

... the result looks as follows ...

\include SparseMatrixExample_Constructor_rowCapacities_vector.out

while the other uses an initializer list ...

\include SparseMatrixExample_Constructor_init_list_1.cpp

... the result looks as follows.

\include SparseMatrixExample_Constructor_init_list_1.out.



### Flexible reduction in matrix rows
### Sparse-matrix vector product
### Sparse matrix IO
### Sparse matrix view

## Tridiagonal matrices <a name="tridiagonal_matrices"></a>

### Dense matrix allocation and initiation
### Flexible reduction in matrix rows
### Dense-matrix vector product
### Dense matrix IO
### Dense matrix view

## Multidiagonal matrices <a name="multidiagonal_matrices"></a>

## Lambda matrices <a name="lambda_matrices"></a>
