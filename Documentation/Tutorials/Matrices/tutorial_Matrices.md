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
* `Index` is a type to be used for indexing of the matrix elements. It `int` by default.
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

Similar to array view (\ref TNL::Containers::ArayView) and vector view (\ref TNL::Containers::VectorView), matrices also offer their view for easier use with lambda functions. For the dense matrix there is a `DenseMatrixView` (\ref TNL::Matrcioes::DenseMatrixView). We will demonstrate it on the example showing the method `setElement` (\ref TNL::Matrices::DenseMatrix::setElement). However, the `SharedPointer` will be replaced with the `DenseMatrixView`. The code looks as follows:

\include DenseMatrixViewExample_setElement.cpp

And the result is:

\include DenseMatrixViewExample_setElement.out


## Sparse matrices <a name="sparse_matrices"></a>

## Tridiagonal matrices <a name="tridiagonal_matrices"></a>

## Multidiagonal matrices <a name="multidiagonal_matrices"></a>

## Lambda matrices <a name="lambda_matrices"></a>
