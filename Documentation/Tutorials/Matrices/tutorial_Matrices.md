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

Dense matrix is a templated class defined in the namespace \ref TNL::Matrices. It has five template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for GPU supporting CUDA. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default it is the row-major order if the matrix is allocated in the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.

### Dense matrix allocation and initiation

The following examples show how to allocate the dense matrix and how to initialize the matrix elements. Small matrices can be created simply by the constructor with an initializer list.

\includelineno Matrices/DenseMatrix/DenseMatrixExample_Constructor_init_list.cpp

In fact, the constructor takes a list of initializer lists. Each embedded list defines one matrix row and so the number of matrix rows is given by the size of the outer initializer list.  The number of matrix columns is given by the longest inner initializer lists. Shorter inner lists are filled with zeros from the right side. The result looks as follows:

\include DenseMatrixExample_Constructor_init_list.out

Larger matrices can be set-up with methods `setElement` and `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement). The following example shows how to call these methods from the host.

\includelineno DenseMatrixExample_addElement.cpp

As we can see, both methods can be called from the host no matter where the matrix is allocated. If it is on GPU, each call of `setElement` or `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement) causes slow transfer of tha data between CPU and GPU. Use this approach only if the performance is not a priority for example for matrices which are set only once this way. The result looks as follows:

\include DenseMatrixExample_addElement.out

More efficient way of the matrix initialization on GPU consists in calling the methods `setElement` and `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement) directly from GPU. It is demonstrated in the following example (of course it works even for CPU):

\includelineno DenseMatrixExample_setElement.cpp

Here we use `SharedPointer` (\ref TNL::Pointers::SharedPointer) to make the matrix accessible in lambda functions even on GPU. We first call the `setElement` method from CPU to set the `i`-th diagonal element to `i`. Next we iterate over the matrix rows with `ParallelFor`and for each row we call a lambda function `f`. This is done on the same device where the matrix is allocated and so it is more efficient for matrices allocated on GPU. In the lambda function we just set the `i`-th diagonal element to `-i`. The result looks as follows:

\include DenseMatrixExample_setElement.out

If we want to set more matrix elements in each row, we can use inner for-loop in the lambda function `f`. This, however, is limiting the parallelization and it can be inefficient for larger matrices. The next example demonstrates a method `forRows` (\ref TNL::Matrices::DenseMatrix::forRows) which iterates over all matrix elements in parallel and it calls a lambda function defining an operation we want to do on the matrix elements.

\includelineno DenseMatrixExample_forRows.cpp

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


\includelineno DenseMatrixExample_rowsReduction_vectorProduct.cpp

The `fetch` lambda function computes the product \f$ a_{ij}x_j\f$ where \f$ a_{ij} \f$ is represented by `value` and \f$x_j \f$ is represented by `xView[columnIdx]`. The reduction is just sum of results particular products and it is represented by by the lambda function `reduce`. Finaly, the lambda function `keep` is responsible for storing the results of reduction in each matrix row (which is represented by the variable `value`) into the output vector `y`.  
The result looks as:

\include DenseMatrixExample_rowsReduction_vectorProduct.out

We will show one more example which is computation of maximal absolute value in each matrix row. The results will be stored in a vector:

\f[
y_i = \max_{j=1,\ldots,n} |a_{ij}|.
\f]

See the following example:

\includelineno DenseMatrixExample_rowsReduction_maxNorm.cpp


The `fetch` lambda function just returns absolute value of \f$a_{ij} \f$ which is represented again by the varibale `value`. The `reduce` lambda function returns larger of given values and the lambda fuction 'keep' stores the results to the output vectro the same way as in the previous example. Of course, if we compute the maximum of all output vector elements we get some kined of max matrix norm. The output looks as:

\include DenseMatrixExample_rowsReduction_maxNorm.out

### Dense matrix-vector product

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

The dense matrix can be saved to a file using a method `save` (\ref TNL::Matrices::DenseMatrix::save) and restored with a method `load` (\ref TNL::Matrices::DenseMatrix::load). To print the matrix, there is a method `print` (\ref TNL::Matrices::DenseMatrix::print) can be used.

### Dense matrix view

Similar to array view (\ref TNL::Containers::ArayView) and vector view (\ref TNL::Containers::VectorView), matrices also offer their view for easier use with lambda functions. For the dense matrix there is a `DenseMatrixView` (\ref TNL::Matrices::DenseMatrixView) which is a templated class with the following template arguments (they are the same as for `DenseMatrix` -- \ref TNL::Matrices::DenseMatrix -- except of the allocator):

* `Real` is a type of matrix elements. 
* `Device` is a device on which the matrix is allocated. This can be \ref TNL::Devices::Host or \ref TNL::Devices::Cuda.
* `Index` is a type for indexing the matrix elements and also row and column indexes.
* `Organization` tells the ordering of matrix elements in memory. It is either RowMajorOrder or ColumnMajorOrder.

The first main reason for using the dense matrix view is its ability to be captured by lambda functions since the copy constructor makes only shallow copy. We will demonstrate it on the example showing the method `setElement` (\ref TNL::Matrices::DenseMatrix::setElement). However, the `SharedPointer` will be replaced with the `DenseMatrixView`. The code looks as follows:

\includelineno DenseMatrixViewExample_setElement.cpp

You can see that we do not need to use the shared pointer (\ref TNL::Pointers::SharedPointer) as we did in the example demonstrating the method `setElement` for dense matrix.  And the result is:

\include DenseMatrixViewExample_setElement.out

The second reason for using the `DenseMatrixView` is to encapsulate data allocated by some other library or program then TNL. The following example demonstrates how to do it:

\includelineno DenseMatrixViewExample_data_encapsulation.cpp

On the lines 18--34 we create matrix by allocating array `data` and filling the matrix using a formula \f$ a_{ij} = i * size + j + 1\f$. We do it first on the host (lines 18--21) in auxilliary array `host_data` to make initiation of the array `data` easier in case when `Device` is GPU. Next, depending on the argument `Device`, we allocate the array `data` on the host or on GPU and copy data from the arary `host_data` to the array `data`. To insert this array into the dense matrix view, we first need to encapsulate it with vector view (\ref TNL::Conatianers::VectorView) `dataView` on the line 39 which can be then used to create the dense matrix view `matrix` on the line 40. Note that wee must set proper matrix elements organizationa which is `RowMajorOrder` (\ref TNL::Algorithms::Segments::RowMajorOrder) in this example. Next, we print the matrix to see if the encapsulation was succesfull (lines 42 and 43) and finaly we demonstrate manipulation with matrix elements (lines 45--48) and we print the result (lines 50 and 51). 

The result looks as follows:

\includelineno DenseMatrixViewExample_data_encapsulation.out

The dense matrix view offers almost all methods which the dense matrix does. So it can be easily used at almost any situation the same way as the dense matrix itself.

## Sparse matrices <a name="sparse_matrices"></a>

[Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) are extremely important in a lot of numerical algorithms. They are used at situations when we need to operate with matrices having majority of the matrix elements equal to zero. In this case, only the non-zero matrix elements are stored with possible some *padding zeros* used for memory alignment. This is necessary mainly on GPUs. Consider just matrix having 50,000 rows and columns whih is 2,500,000,000 matrix elements. If we store each matrix element in double precision (it means eight bytes per element) we need 20,000,000,000 bytes which is nearly 20 GB of memory. If there are only five non-zero elements in each row we need only \f$8 \times 5 \times 50,000=2,000,000\f$ bytes and so nearly 200 MB. It is really great difference.

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

**If `Real` is set to `bool`, we get *a binary matrix* for which the non-zero elements can be equal only to one and so the matrix elements values are not stored explicitly in the memory.**

### Sparse matrix allocation and initiation

Small matrices can be initialized by a constructor with initializer list. We assume having the following sparse matrix

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

\includelineno SparseMatrixExample_Constructor_init_list_2.cpp

The constructor accepts the following parameters:

* `rows` is a number of matrix rows.
* `columns` is a number of matrix columns.
* `data` is definition of non-zero matrix elements. It is a initializer list of triples having a form `{ row_index, column_index, value }`. In fact, it is very much like the coordinates format - [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)).

The constructor also accepts `Real` and `Index` allocators (\ref TNL::Allocators) but their are not important for this example. A method `setElements` works the same way:

\includelineno SparseMatrixExample_setElements.cpp

The result of both examples looks as follows:

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


\includelineno SparseMatrixExample_setRowCapacities.cpp

The method \ref TNL::Matrices::SparseMatrix::setRowCapacities reads the required capacities of the matrix rows from a vector (or simmilar container - \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector and \ref TNL::Containers::VectorView) which has the same number of elements as the number of matrix rows and each element defines the capacity of the related row. The result looks as follows:

\include SparseMatrixExample_setRowCapacities.out

There are constructors which also set the row capacities. The first one uses a vector:

\includelineno SparseMatrixExample_Constructor_rowCapacities_vector.cpp

The second one uses an initializer list:

\includelineno SparseMatrixExample_Constructor_init_list_1.cpp

The result of both examples looks as follows:

\include SparseMatrixExample_Constructor_init_list_1.out

Finaly, there is a constructor which creates the sparse matrix from 'std::map'. It is usefull especially in situation when you cannot compute the matrix elements by rows but rather in random order. You can do it on CPU and store the matrix elements in `std::map` data structure in a [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) format manner. It means that each entry of the `map` is the following pair:

```
std::pair( std::pair( row_index, column_index ), element_value )
```

which defines one matrix element at given coordinates with given value. Of course, you can insert such entries in any order into the `map`. When it is complete you can pass it the sparse matrix. See the following example:

\includelineno SparseMatrixExample_Constructor_std_map.cpp

A method `setElements` works the same way for already existing instances of sparse matrix:


\includelineno SparseMatrixExample_setElements_map.cpp

The result of both examples looks as folows:

\include SparseMatrixExample_setElements_map.out

Another way of setting the sparse matrix is via the methods `setElement` and `addElement` (\ref TNL::Matrices::SparseMatrix::setElement, \ref TNL::Matrices::addElement). The procedure is as follows:

1. Setup the matrix dimensions.
2. Setup the row capacities.
3. Setup the matrix elements.

The method can be called from both host (CPU) and device (GPU) if the matrix is allocated there. Note, however, that if the matrix is allocated on GPU and the method is called from CPU there will be significant performance drop because the matrix elements will be transfered one after another. However, if the matrix elements setup is not a critical part of your algorithm this can be an easy way how to do it. See the following example:

\includelineno SparseMatrixExample_setElement.cpp

Note that we use `SharedPointer` (\ref TNL::Pointers::SharedPointer) to pass the matrix easily into the lambda function when it runs on GPU. The first for-loop runs on CPU no matter where the matrix is allocated. Next we call the lambda function `f` from `ParallelFor` which is device sensitive and so it runs on CPU or GPU depending where the matrix is allocated. To avoid use of `SharedPointer`, which requires explicit synchronization of smart pointers, you may use `SparseMatrixView' (\ref TNL::Matrices::SparseMatrixView) to achiev the same. The result looks as follows:

\include SparseMatrixExample_setElement.out

The method `addElement` adds a value to specific matrix element. Otherwise, it behaves the same as `setElement`. See the following example:

\includelineno SparseMatrixExample_addElement.cpp

The result looks as follows:

\include SparseMatrixExample_addElement.out

Finaly, for the most efficient way of setting the non-zero matrix elements, is use of a method `forRows`. It requires indexes of the range of rows (`begin` and `end`) to be processed and a lambda function `function` which is called for each non-zero element. The lambda functions provides the following data:

* `rowIdx` is a row index of the matrix element.
* `localIdx` is an index of the non-zero matrix element within the matrix row.
* `columnIdx` is a column index of the matrix element. If the matrix element is suppsoed to be changed, this parameter can be a reference and so its value can be changed.
* `value` is a value of the matrix element. It the matrix element is supposed to be changed, this parameter can be a reference as well and so the element value can be changed.
* `compute` is a bool reference. When it is set to `false` the rest of the row can be omitted. This is, however, only a hint and it depends on the underlying matrix format if it is taken into account.

See the following example:

\includelineno SparseMatrixExample_forRows.cpp

On the line 9, we allocate a lower triangular matrix (because the row capacities `{1,2,3,4,5}` are equal to row index) using the `SparseMatrix`. On the line 11, we prepare lambda function `f` which we execute on the line 22 just by calling the method `forRows` (\ref TNL::Matrices::SpartseMatrix::forRows). This method takes the range of matrix rows as the first two parameters and the lambda function as the last parameter. The lambda function receives parameters metioned above (see the line 11). We first check if the matrix element coordinates (`rowIdx` and `localIdx`) points to an element lying before the matrix diagonal or on the diagonal. At this moment we should better explain the meaning of the parameter `localIdx`. It says the local index or the range of the non-zero element in the matrix row. The sparse matrix formats usualy in the first step compress the matrix rows by omitting the zero matrix elements as follows

\f[
\left(
\begin{array}{ccccc}
0 & 1 & 0 & 2 & 0 \\
0 & 0 & 5 & 0 & 0 \\
4 & 0 & 0 & 0 & 7 \\
0 & 3 & 0 & 8 & 5 \\
0 & 5 & 7 & 0 & 0
\end{array}
\right)
\rightarrow
\left(
\begin{array}{ccccc}
1 & 2 & . & . & . \\
5 & . & . & . & . \\
4 & 7 & . & . & . \\
3 & 8 & 5 & . & . \\
5 & 7 & . & . & .
\end{array}
\right)
\f]

Some sparse matrix formats adds back padding zeros for better alignment of data in memory. But if this is not the case, the local indexes of the matrix elements would read as:

\f[
\left(
\begin{array}{ccccc}
0 & 1 & . & . & . \\
0 & . & . & . & . \\
0 & 1 & . & . & . \\
0 & 1 & 2 & . & . \\
0 & 1 & . & . & .
\end{array}
\right)
\f]

In case of the lower triangular matrix in our example, the local index is in fact the same as the column index

\f[
\left(
\begin{array}{ccccc}
0 & . & . & . & . \\
0 & 1 & . & . & . \\
0 & 1 & 2 & . & . \\
0 & 1 & 2 & 3 & . \\
0 & 1 & 2 & 3 & 4
\end{array}
\right)
\f]

If we call the method `forRows` to setup the matrix elements for the first time, the parameter `columnIdx` has no sense because the matrix elements and their column indexes were not set yet. Therefore it is important that the test on the line 12 reads as

```
if( rowIdx < localIdx )
```

because

```
if( rowIdx < columnIdx )
```

would not make sense. If we pass through this test, the matrix element lies in the lower triangular part of the matrix and we may set the matrix elements which is done on the lines 17 and 18. The column index (`columnIdx`) is set to local index (line 17) and `value` is set on the line 18. The result looks as follows:


\includelineno SparseMatrixExample_forRows.out

### Flexible reduction in matrix rows

The *flexible parallel reduction* in rows for sparse matrices is very simmilar to the one for dense matrices. It consits of three lambda functions:

1. `fetch` reads and preproces data entering the flexible parallel reduction.
2. `reduce` performs the reduction operation.
3. `keep` stores the results from each matrix row.

See the following example:

\includelineno SparseMatrixExample_rowsReduction_vectorProduct.cpp

On the lines 11-16 we set the following matrix:

\f[
\left(
\begin{array}{ccccc}
1 & . & . & . & . \\
1 & 2 & . & . & . \\
. & 1 & 8 & . & . \\
. & . & 1 & 9 & . \\
. & . & . & . & 1
\end{array}
\right)
\f]

Next we prepare input (`x`) and output (`y`) vectors on the lines 21 and 22 and set all elements of the input vector to one (line 27). Since we will need to access these vectors in lambda functions we prepare their views on lines 32 and 33. On the lines 39-41, we define the `fetch` lambda function. It receives three arguments:

1. `rowIdx` is a row index of the matrix element being currently processed.
2. `columnIdx` is a column index of the matrix elements being currently processed.
3. `value` is a value of the matrix element being currently procesed.

We ommit the row index and take the column index which indicates index of the element of the input vector we need to fetch (`xView[ columnIdx ]`). We take its value and multiply it with the value (`value`) of the current matrix element. We do not need to write lambda function for reduction since it is only summation of the intermediate results from the `fetch` lamda and we can use `std::plus<>{}` (see the line 60). The `keep` lambda function offers two parameters:

1. `rowIdx` tells the index of the matrix row for which we aim to store the result.
2. `value` is the result obtained in the given matrix row.

In our example, we just write the result into appropriate element of the output vector `y` which is given just by the row index `rowIdx` -- see the line 47.  On the line 53 we start the computation of the matrix-vector product. The method `rowsReduction` (\ref TNL::Matrices::SparseMatrix::rowsReduction) accepts the following arguments:

1. `begin` is the begining of the matrix rows range on which the reduction will be performed.
2. `end` is the end of the matrix rows range on which the reduction will be performed. The last matrix row which is going to be processed has index `end-1`.
3. `fetch` is the fetch lambda function.
4. `reduce` is the the lmabda function performing the reduction.
5. `keep` is the lambda function responsible for processing the results from particular matrix rows.
6. `zero` is the "zero" element of given reduction opertation also known as *idempotent*. It is really 0 for summation in our example (adding zero to any number does not change the result).

At the end we print the matrix, the input and the output vector -- lines 55-57. The result looks as follows:

\include SparseMatrixExample_rowsReduction_vectorProduct.out

### Sparse matrix-vector product

As we mentioned already in the part explaining the dense matrices, matrix-vector multiplication or in this case sparse matrix-vector multiplication ([SpMV](https://en.wikipedia.org/wiki/Sparse_matrix-vector_multiplication)) is one of the most important operations in numerical mathematics and high-performance computing. It is represented by a method `vectorProduct` (\ref TNL::Matrices::SparseMatrix::vectorProduct). It is templated method with two template parameters `InVector` and `OutVector` telling the types of input and output vector respectively. Usually one will substitute some of \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView for these types. The method computes the following formula

```
outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector
```

and it accepts the following parameters:

* `inVector` is the input vector having the same number of elements as the number of matrix columns.
* `outVector` is the output vector having the same number of elements as the number of matrix rows.
* `matrixMultiplicator` is a number by which the result of matrix-vector product is multiplied. 
* `outVectorMultiplicator` is a number by which the output vector is multiplied before added to the result of matrix-vector product.
* `begin` is an index of the first matrix row that is involved in the multiplication. It is zero be default.
* `end` is an index of the last matrix row that is involved in the multiplication. It is the last matrix row by default.

Note that the ouput vector dimension must be the same as the number of matrix rows no matter how we set `begin` and `end` parameters. These parameters just say that some matrix rows and the output vector elements are omitted.

### Sparse matrix IO

The sparse matrix can be saved to a file using a method `save` (\ref TNL::Matrices::SparseMatrix::save) and restored with a method `load` (\ref TNL::Matrices::SparseMatrix::load). For printing the matrix, there is a method `print` (\ref TNL::Matrices::SparseMatrix::print) can be used.

### Sparse matrix view

Sparse matrix view serves, simillar to other views in TNL, to data sharing and for use with lambda functions (views can be easily captured since they make only shallow copy). The sparse matrix view (\ref TNL::Matrices::SparseMatrixView) is templated class having the following template arguments (they are the same as for `SparseMatrix` -- \ref TNL::Matrices::SparseMatrix -- except of the allocators):

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

**If `Real` is set to `bool`, we get *a binary matrix view*.**

The following example shows the use of `SparseMatrixView` with lambda functions:

\includelineno SparseMatrixViewExample_setElement.cpp

The result looks as follows:

\includelineno SparseMatrixViewExample_setElement.out

## Tridiagonal matrices <a name="tridiagonal_matrices"></a>

Tridiagonal matrix format serves for specific matrix pattern when the nonzero matrix elements can be placed only at the diagonal and immediately next to the diagonal. Here is an example:

\f[
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2
 \end{array}
 \right)
\f]

An advantage is that we do not store the column indexes  explicitly as it is in \ref TNL::Matrices::SparseMatrix. This can reduce significantly the  memory requirements which also means better performance. See the following table for the storage requirements comparison between \ref TNL::Matrices::TridiagonalMatrix and \ref TNL::Matrices::SparseMatrix.

 
  Real   | Index      |      SparseMatrix    | TridiagonalMatrix   | Ratio
 --------|------------|----------------------|---------------------|--------
  float  | 32-bit int | 8 bytes per element  | 4 bytes per element | 50%
  double | 32-bit int | 12 bytes per element | 8 bytes per element | 75%
  float  | 64-bit int | 12 bytes per element | 4 bytes per element | 30%
  double | 64-bit int | 16 bytes per element | 8 bytes per element | 50%
 
Tridiagonal matrix is a templated class defined in the namespace \ref TNL::Matrices. It has five template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for GPU supporting CUDA. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default it is the row-major order if the matrix is allocated in the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.

### Tridiagonal matrix allocation and initiation

The tridiagonal matrix can be initialized by the means of the constructor with initializer list. The matrix from the begining of this section can be constructed as the following example shows:

\includelineno TridiagonalMatrixExample_Constructor_init_list_1.cpp

For better alignment in the memory the tridiagonal format is organised like if there were three nonzero matrix elements in each row. This is not true for example in the first row where there is no matrix element on the left side of the diagonal. The same happens on the last row of the matrix. In our example, we have to add even the artificial matrix elements like this:

\f[
\begin{array}{c}
0 \\
. \\
. \\
. \\
. \\
.
\end{array}
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2
 \end{array}
 \right)
 \begin{array}{c}
. \\
. \\
. \\
. \\
. \\
0
\end{array}
\f]

If a matrix has more rows then columns, we have to extend the last two rows with nonzero elements in this way

\f[
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2 \\
  .  &  .  &  .  &  .  &  . & -1
 \end{array}
 \right)
\rightarrow
\begin{array}{c}
0 \\
. \\
. \\
. \\
. \\
. \\
.
\end{array}
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2 \\
  .  &  .  &  .  &  .  &  . & -1
 \end{array}
 \right)
 \begin{array}{cc}
. & . \\
. & . \\
. & . \\
. & . \\
. & . \\
0 & . \\
0 & 0
\end{array}
\f]

The output of the example looks as:

\includelineno TridiagonalMatrixExample_Constructor_init_list_1.out

Similar way of the tridiagonal matrix setup is offered by the method `setElements` (\ref TNL::Matrices::TridiagonalMatrix::setElements) as the following example demonstrates:

\includelineno TridiagonalMatrixExample_setElements.cpp


Here we create the matrix in two steps. Firstly, we setup the matrix dimensions by the appropriate constructor (line 24) and after that we setup the matrix elements (line 25-45). The result looks the same as in the previous example:

\includelineno TridiagonalMatrixExample_setElements.out

In the following example we create tridiagonal matrix with 5 rows and 5 columns (line 12-14) by the means of a shared pointer (\ref TNL::Pointers::SharedPointer) to make this work even on GPU. We set numbers 0,...,4 on the diagonal (line 16) and we print the matrix (line 18). Next we use a lambda function (lines 21-27) combined with parallel for (\ref TNL::Algorithms::ParallelFor) (line 35), to modify the matrix. The offdiagonal elements are set to 1 (lines 23 and 26) and for the diagonal elements, we change the sign (line 24).

\includelineno TridiagonalMatrixExample_setElement.cpp

The result looks as follows:

\includelineno TridiagonalMatrixExample_setElement.out

 A slightly simpler way how to do the same with no need for shared pointer (\ref TNL::Pointers::SharedPointer), could be with the use of tridiagonal matrix view and the method `getRow` (\ref TNL::Matrices::TridiagonalMatrixView::getRow) as the following example demonstrates:

\includelineno TridiagonalMatrixViewExample_getRow.cpp

We create a matrix with the same size (line 10-15) set ones on the diagonal (lines 15-16). Next, we fetch the tridiagonal matrix view (line 16) which we can refer in the lambda function for matrix elements modification (lines 18-26). Inside the lambda function, we first get a matrix row by calling the method `getRow` (\ref TNL::Matrices::TridiagonalMatrixView::getRow) using which we can acces the matrix elements (lines 21-25). The lambda function is called by the parallel for (\ref TNL::Algorithms::ParallelFor).

The result looks as follows:

\includelineno TridiagonalMatrixViewExample_getRow.out

Finaly, even a bit more simple and bit less flexible way of matrix elements manipulation with use of the method `forRows` (\ref TNL::Matrices::TridiagonalMatrix::forRows) is demosntrated in the following example:

\includelineno TridiagonalMatrixViewExample_forRows.cpp

On the line 41 we call the method `forRows` (\ref TNL::Matrices::TridiagonalMatrix::forRows) instead of parallel for (\ref TNL::Algorithms::ParallelFor). This method iterates overl all matrix rows and all nonzero matrix elements. The lambda function function on the line 24 therefore do not receive only the matrix row index but also local index of the matrix element (`localIdx`) which is a rank of the nonzero matrix element in given row. The values of the local index for given matrix elements is as follows

\f[ 
\left(
\begin{array}{cccccc}
1 & 2 &   &   &   &     \\
0 & 1 & 2 &   &   &     \\
  & 0 & 1 & 2 &   &     \\
  &   & 0 & 1 & 2 &     \\
  &   &   & 0 & 1 & 2   \\
  &   &   &   & 0 & 1
\end{array}
\right)
\f]

Next parameter `columnIdx` received by the lambda function is the column index of the matrix element. The fourth parameter `value` is a reference on the matrix element which we use for its modification. If the last parameter `compute` is set to false, the iterations over the matrix rows is terminated.

The result looks as follows:

\includelineno TridiagonalMatrixViewExample_forRows.out

### Flexible reduction in matrix rows

The *flexible parallel reduction* in rows for tridiagonal matrices is also simmilar as for dense and sparse matrices. It is represented by three lambda functions:

1. `fetch` reads and preproces data entering the flexible parallel reduction.
2. `reduce` performs the reduction operation.
3. `keep` stores the results from each matrix row.

See the following example:

\includelineno TridiagonalMatrixExample_rowsReduction.cpp

Here we first set tridiagonal matrix (lines 10-27) which looks as

\f[
\left(
\begin{array}{ccccc}
1 & 3 &   &   &   &    \\
2 & 1 & 3 &   &   &    \\
  & 2 & 1 & 3 &   &    \\
  &   & 2 & 1 & 3 &    \\
  &   &   & 2 & 1 & 3
\end{array}
\right).
\f]

Next we want to compute maximal absolute value of the nonzero matrix elements in each row. We allocate the vector `rowMax` where we will store the results (line 32). The lambda function `fetch` (lines 42-44) is responsible for reading the matrix elements. It receives three arguments:

1. `rowIdx` is a row index of the matrix element being currently processed.
2. `columnIdx` is a column index of the matrix elements being currently processed.
3. `value` is a value of the matrix element being currently procesed.

In our example, the only thing this function has to do, is to compute the absolute value of each matrix element represented by variable `value`. The next lambda function, `reduce` (lines 49-51), performs reduction operation. In this case, it returns maximum of two input values `a` and `b`. Finaly, the lambda function `keep` (lines 56-58) is defined with the following parameters:

1. `rowIdx` tells the index of the matrix row for which we aim to store the result.
2. `value` is the result obtained in the given matrix row.

In our example, it just takes the result of the reduction in variable `value` in each row and stores it into the vector `rowMax` via related vector view `rowMaxView`.

The method `rowsReduction` (\ref TNL::Matrices::SparseMatrix::rowsReduction) activates all the mantioned lambda functions (line 63). It accepts the following arguments:

1. `begin` is the begining of the matrix rows range on which the reduction will be performed.
2. `end` is the end of the matrix rows range on which the reduction will be performed. The last matrix row which is going to be processed has index `end-1`.
3. `fetch` is the fetch lambda function.
4. `reduce` is the the lmabda function performing the reduction.
5. `keep` is the lambda function responsible for processing the results from particular matrix rows.
6. `zero` is the "zero" element of given reduction opertation also known as *idempotent*. In our example, the role of this element has the lowest number of given type which we can obtain using function `std::numeric_limits< double >::lowest()` from STL.

 The results looks as follows:

\includelineno TridiagonalMatrixExample_rowsReduction.out

### Tridiagonal matrix-vector product

Similar to dense and sparse matrices, matrix-vector multiplication is represented by a method `vectorProduct` (\ref TNL::Matrices::TridiagonalMatrix::vectorProduct). It is templated method with two template parameters `InVector` and `OutVector` telling the types of input and output vector respectively. Usually one will substitute some of \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView for these types. The method computes the following formula

```
outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector
```

and it accepts the following parameters:

* `inVector` is the input vector having the same number of elements as the number of matrix columns.
* `outVector` is the output vector having the same number of elements as the number of matrix rows.
* `matrixMultiplicator` is a number by which the result of matrix-vector product is multiplied. 
* `outVectorMultiplicator` is a number by which the output vector is multiplied before added to the result of matrix-vector product.
* `begin` is an index of the first matrix row that is involved in the multiplication. It is zero be default.
* `end` is an index indicating the last matrix row that is involved in the multiplication which is `end - 1`. It is the number of matrix rows.

Note that the output vector dimension must be the same as the number of matrix rows no matter how we set `begin` and `end` parameters. These parameters just say that some matrix rows and the output vector elements are omitted.

### Tridiagonal matrix IO

The tridiagonal matrix can be saved to a file using a method `save` (\ref TNL::Matrices::TridiagonalMatrix::save) and restored with a method `load` (\ref TNL::Matrices::TridiagonalMatrix::load). For printing the matrix, there is a method `print` (\ref TNL::Matrices::TridiagonalMatrix::print) can be used.

### Tridiagonal matrix view

Similar to dense and sparse matrix view, tridiagonal matrix also offers its view for easier use with lambda functions. It is represented by a templated class \ref TNL::Matrices::TridiagonalMatrixView with the following template parameters:

* `Real` is a type of matrix elements. 
* `Device` is a device on which the matrix is allocated. This can be \ref TNL::Devices::Host or \ref TNL::Devices::Cuda.
* `Index` is a type for indexing the matrix elements and also row and column indexes.
* `Organization` tells the ordering of matrix elements in memory. It is either RowMajorOrder or ColumnMajorOrder.

The first main reason for using the matrix view is its ability to be captured by lambda functions since the copy constructor makes only shallow copy. We can demonstrate it on the example showing the method `setElement` (\ref TNL::Matrices::TridiagonalMatrix::setElement). The code looks as follows:

\includelineno TridiagonalMatrixViewExample_setElement.cpp

The matrix view is obtained by the method `getView` (\ref TNL::Matrices::TridiagonalMatrix::getView) on the line 13. We firsrt show, that the view can be used the same way as common matrix (lines 14 and 15) but it can be used the same way even in lambda functions as we can see on the lines 20-26. Compare it with the same example using shared pointer instead of the matrix view:

\includelineno TridiagonalMatrixExample_setElement.cpp

The main disadventages are:

1. The shared pointer must be created together with the matrix (line 14) and there is no way to get it later. The matrix view can be obtained from any matrix at any time.
2. We have to synchronize shared pointers explicitly by calling the function \ref TNL::Pointers::synchronizeSmartPointersOnDevice (line 34).

So for the sake of using a matrix in lambda functions, the matrix view is better tool. The result of both examples looks as:

\include TridiagonalMatrixExample_setElement.out

As we mentioned already, the tridiagonal matrix view offers almost all methods which the tridiagonal matrix does. So it can be easily used at almost any situation the same way as the tridiagonal matrix itself.

## Multidiagonal matrices <a name="multidiagonal_matrices"></a>

Multidiagonal matrices are generalization of the tridiagonal matrix. It is a special type of sparse matrices with specific pattern of the nonzero matrix elements which are positioned only parallel along diagonal. See the following example:

\f[
  \left(
  \begin{array}{ccccccc}
   4  & -1  &  .  & -1  &  . & .  \\
  -1  &  4  & -1  &  .  & -1 & .  \\
   .  & -1  &  4  & -1  &  . & -1 \\
  -1  & .   & -1  &  4  & -1 &  . \\
   .  & -1  &  .  & -1  &  4 & -1 \\
   .  &  .  & -1  &  .  & -1 &  4
  \end{array}
  \right)
 \f]

 We can see that the matrix elements lay on lines parallel to the main diagonal. Such lines can be expressed by their offsets from the main diagonal. On the following figure, each such line is depicted in different color:

  \f[
\begin{array}{ccc}
\color{green}{-3} & .                & \color{cyan}{-1} \\
\hline
 \color{green}{*} & .                & \color{cyan}{*} \\
 .                & \color{green}{*} & . \\
 .                & .                & \color{green}{*} \\
 .                & .                & . \\
 .                & .                & . \\
 .                & .                & . 
\end{array}
\left(
  \begin{array}{ccccccc}
 \color{blue}{0}    & \color{magenta}{1}   & .                   & \color{red}{3}      & .                   & . \\
   \hline
  \color{blue}{4}   & \color{magenta}{-1}  &  .                  & \color{red}{-1}     &  .                  & .  \\
  \color{cyan}{-1}  & \color{blue}{4}      & \color{magenta}{-1} &  .                  & \color{red}{-1}     & .  \\
   .                & \color{cyan}{-1}     & \color{blue}{4}     & \color{magenta}{-1} &  .                  & \color{red}{-1} \\
  \color{green}{-1} & .                    & \color{cyan}{-1}    & \color{blue}{4}     & \color{magenta}{-1} &  . \\
   .                & \color{green}{-1}    &  .                  & \color{cyan}{-1}    &  \color{blue}{4}    & \color{magenta}{-1} \\
   .                &  .                   & \color{green}{-1}   &  .                  & \color{cyan}{-1}    &  \color{blue}{4}
  \end{array}
  \right)
 \f]

 In this matrix, the offsets reads as \f$\{-3, -1, 0, +1, +3\}\f$. It also means that the column indexes on \f$i-\f$th row are \f$\{i-3, i-1, i, i+1, i+3\}\f$ (where the resulting index is non-negative and  smaller than the number of matrix columns). An advantage is that, similar to the tridiagonal matrix (\ref TNL::Matrices::TridiagonalMatrix), we do not store the column indexes explicitly as it is in \ref SparseMatrix. This can reduce significantly the  memory requirements which also means better performance. See the following table for the storage requirements comparison between \ref TNL::Matrices::MultidiagonalMatrix and \ref TNL::Matrices::SparseMatrix.

  Real   | Index     |      SparseMatrix    | MultidiagonalMatrix | Ratio
 --------|-----------|----------------------|---------------------|--------
  float  | 32-bit int| 8 bytes per element  | 4 bytes per element | 50%
  double | 32-bit int| 12 bytes per element | 8 bytes per element | 75%
  float  | 64-bit int| 12 bytes per element | 4 bytes per element | 30%
  double | 64-bit int| 16 bytes per element | 8 bytes per element | 50%
 
Multidiagonal matrix is a templated class defined in the namespace \ref TNL::Matrices. It has six template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for GPU supporting CUDA. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default it is the row-major order if the matrix is allocated in the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.
* `IndexAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements offsets. By default, it is the default allocator for given `Index` type and `Device` type -- see \ref TNL::Allocators::Default.

### Multidiagonal matrix allocation and initiation

The construction of the multidiagonal matrix differs from the tridiagonal mainly in necessity to define the offsets of "subdiagonals" as we demonstrate on the following example which creates matrix like of the following form:

\f[
\left(
\begin{array}{cccccccccccccccc}
1  &  . &    &    &  . &    &    &    &    &    &    &    &     &    &    &   \\
.  &  1 &  . &    &    &  . &    &    &    &    &    &    &     &    &    &   \\
   &  . &  1 &  . &    &    & .  &    &    &    &    &    &     &    &    &   \\
   &    &  . &  1 &  . &    &    &  . &    &    &    &    &     &    &    &   \\
.  &    &    &  . &  1 & .  &    &    & .  &    &    &    &     &    &    &   \\
   & -1 &    &    & -1 & 1  & -1 &    &    & -1 &    &    &     &    &    &   \\
   &    & -1 &    &    & -1 &  1 & -1 &    &    & -1 &    &     &    &    &   \\
   &    &    & .  &    &    &  . &  1 & .  &    &    & .  &     &    &    &   \\
   &    &    &    & .  &    &    &  . & 1  &  . &    &    &  .  &    &    &   \\
   &    &    &    &    & -1 &    &    & -1 &  1 & -1 &    &     & -1 &    &   \\
   &    &    &    &    &    & -1 &    &    & -1 &  1 & -1 &     &    & -1 &   \\
   &    &    &    &    &    &    &  . &    &    &  . &  1 &  .  &    &    & . \\
   &    &    &    &    &    &    &    & .  &    &    &  . &  1  & .  &    &   \\
   &    &    &    &    &    &    &    &    &  . &    &    &  .  & 1  & .  &   \\
   &    &    &    &    &    &    &    &    &    &  . &    &     & .  & 1  & . \\
   &    &    &    &    &    &    &    &    &    &    & .  &     &    & .  & 1
\end{array}
\right)
\f]

The code reads as:

\includelineno MultidiagonalMatrixExample_Constructor.cpp

The matrix from this example arises from a discretization of the [Laplace operator in 2D by the finite difference method](https://en.wikipedia.org/wiki/Discrete_Poisson_equation). We use this example because it is very frequent numerical problem. If the reader, however, is not familiar with the finite difference method, please, do not be scared, we will just create the matrix mentioned above.

We firstly compute the matrix size (`matrixSize`) based on the numerical grid dimensions on the line 16. The subdiagonals offsets are defined by the numerical grid size and since it is four in this example the offsets read as \f$\left\{-4,-1,0,1,4 \right\} \f$ or `{ -gridSize, -1, 0, 1, gridSize}` (line 17). Here we store the offsets (referred as `shifts`) in vector (\ref TNL::Containers::Vector). Next we use a constructor with matrix dimensions and offsets passed via TNL vector (line 18). Next we fetch matrix view (line 19) (see [Multidiagonal matrix view](#multidiagonal_matrix_view)).

The matrix is constructed by iterating over particular nodes of the numerical grid. Each node corresponed to one matrix row. This is why the lambda function `f` (lines 20-35) take two indexes `i` and `j` (line 20). Their values are coordinates of the twodimensional numerical grid. Based on these coodrinates we compute index (`elementIdx`) of the corresponding matrix row (line 21). We fetch matrix row (`row`) by calling the `getRow` method (\ref TNL::Matrices::MutlidiagonalMatrix::getRow) (line 22). Depending on the grid node coordinates we set either the boundary conditions (lines 23-26) for the boundary nodes (those laying on the boundary of the grid and so their coordinates fulfil the condition `i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1` ) for which se set onle diagonal element to 1. The inner nodes of the numerical grid are handled on the lines 29-33 where we set coefficients approximating the Laplace operator. We use the method `setElement` of the matrix row (\ref TNL::Matrices::MultidiagonalMatrixRow::setElement) which takes the local index of the nonzero matrix element as the first parametr and the new value of the element as the second parameter. The local indexes, in fact, refer to particular subdiagonals as depicted on the following figure (in blue): 

\f[
\begin{array}{cccc}
\color{blue}{-4} &   &   & \color{blue}{-1} \\
\hline
.  &   &   & .  \\
   & . &   &    \\
   &   & . &    \\
   &   &   & .  \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &   
\end{array}
\left(
\begin{array}{cccccccccccccccc}
\color{blue}{0}  &  \color{blue}{1} &    &    &  \color{blue}{4} &    &    &    &    &    &    &    &     &    &    &   \\
\hline
1  &  . &    &    &  . &    &    &    &    &    &    &    &     &    &    &   \\
.  &  1 &  . &    &    &  . &    &    &    &    &    &    &     &    &    &   \\
   &  . &  1 &  . &    &    & .  &    &    &    &    &    &     &    &    &   \\
   &    &  . &  1 &  . &    &    &  . &    &    &    &    &     &    &    &   \\
.  &    &    &  . &  1 & .  &    &    & .  &    &    &    &     &    &    &   \\
   & -1 &    &    & -1 & 1  & -1 &    &    & -1 &    &    &     &    &    &   \\
   &    & -1 &    &    & -1 &  1 & -1 &    &    & -1 &    &     &    &    &   \\
   &    &    & .  &    &    &  . &  1 & .  &    &    & .  &     &    &    &   \\
   &    &    &    & .  &    &    &  . & 1  &  . &    &    &  .  &    &    &   \\
   &    &    &    &    & -1 &    &    & -1 &  1 & -1 &    &     & -1 &    &   \\
   &    &    &    &    &    & -1 &    &    & -1 &  1 & -1 &     &    & -1 &   \\
   &    &    &    &    &    &    &  . &    &    &  . &  1 &  .  &    &    & . \\
   &    &    &    &    &    &    &    & .  &    &    &  . &  1  & .  &    &   \\
   &    &    &    &    &    &    &    &    &  . &    &    &  .  & 1  & .  &   \\
   &    &    &    &    &    &    &    &    &    &  . &    &     & .  & 1  & . \\
   &    &    &    &    &    &    &    &    &    &    & .  &     &    & .  & 1
\end{array}
\right)
\f]

We use `ParallelFor2D` (\ref TNL::Algorithms::ParallelFor2D) to iterate over all nodes of the numerical grid (line 36) and apply the lambda function. Also note that for the sake of better memory alignemnt and faster acces to the matrix elements, we store all subdiagonals in complete form including the elemenets which are outside the matrix as depicted on the following figure where zeros stand for the padding artificial zero matrix elements

\f[
\begin{array}{cccc}
0  &   &   & 0  \\
   & 0 &   &    \\
   &   & 0 &    \\
   &   &   & 0  \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &   
\end{array}
\left(
\begin{array}{cccccccccccccccc}
1  &  0 &    &    &  0 &    &    &    &    &    &    &    &     &    &    &   \\
0  &  1 &  0 &    &    &  0 &    &    &    &    &    &    &     &    &    &   \\
   &  0 &  1 &  0 &    &    & 0  &    &    &    &    &    &     &    &    &   \\
   &    &  0 &  1 &  0 &    &    &  0 &    &    &    &    &     &    &    &   \\
0  &    &    &  0 &  1 & 0  &    &    & 0  &    &    &    &     &    &    &   \\
   & -1 &    &    & -1 & 1  & -1 &    &    & -1 &    &    &     &    &    &   \\
   &    & -1 &    &    & -1 &  1 & -1 &    &    & -1 &    &     &    &    &   \\
   &    &    & 0  &    &    &  0 &  1 & 0  &    &    & 0  &     &    &    &   \\
   &    &    &    & 0  &    &    &  0 & 1  &  0 &    &    &  0  &    &    &   \\
   &    &    &    &    & -1 &    &    & -1 &  1 & -1 &    &     & -1 &    &   \\
   &    &    &    &    &    & -1 &    &    & -1 &  1 & -1 &     &    & -1 &   \\
   &    &    &    &    &    &    &  0 &    &    &  0 &  1 &  0  &    &    & 0 \\
   &    &    &    &    &    &    &    & 0  &    &    &  0 &  1  & 0  &    &   \\
   &    &    &    &    &    &    &    &    &  0 &    &    &  0  & 1  & 0  &   \\
   &    &    &    &    &    &    &    &    &    &  0 &    &     & 0  & 1  & 0 \\
   &    &    &    &    &    &    &    &    &    &    & 0  &     &    & 0  & 1
\end{array}
\right)
\begin{array}
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
0  &   &   &    \\
   & 0 &   &    \\
   &   & 0 &    \\
0  &   &   & 0  
\end{array}
\f]


The result looks as follows:

\includelineno MultidiagonalMatrixExample_Constructor.out

Slightly simpler way of doing the same is by using the constructor of multidiagonal matrix taking the subdiagonals offsets as an STL initializer list:

\includelineno MultidiagonalMatrixExample_Constructor_init_list_1.cpp

The only change is on the line 17 which reads as

```
TNL::Matrices::MultidiagonalMatrix< double, Device > matrix( matrixSize, matrixSize, { - gridSize, -1, 0, 1, gridSize } );
```

Here we call the mentioned cosntructor, which accepts the matrix dimensions (number of rows and columns) as first two parameters and the initializer list with the subdiagonal offsets as the last one. The result looks the same as in the previous example.

There is also a constructor with initializer list for matrix elements values as demonstrated by the following example:

\includelineno MultidiagonalMatrixExample_Constructor_init_list_2.cpp

Here, we create a matrix which looks as 

\f[
\left(
\begin{array}{cccccc}
4  & -1 &    & -1 &    &    \\
-1 &  4 & -1 &    & -1 &    \\
   & -1 & 4  & -1 &    & -1 \\
-1 &    & -1 &  4 & -1 &    \\
   & -1 &    & -1 & 4  & -1 \\
   &    & -1 &    & -1 &  4 \\
\end{array}
\right).
\f]

On the lines 25-46, we call the constructor which, in addition to matrix dimensions and subdiagonals offsets, accepts also initializer list of initializer lists with matrix elements values. Each embeded list corresponds to one matrix row and it contains values of matrix elements on particular subdiagonals including those which lies out of the matrix. The resuls looks as follows:

\includelineno MultidiagonalMatrixExample_Constructor_init_list_2.out

The matrix elements values can be changed the same way using the method method `setElements` (\ref TNL::Matrices::MutlidiagonalMatrix::setElements) which accepts the elements values in the same form of embedded initializer list. It just does not allow changing the subdiagonals offsets. For this purpose method `setDiagonalsOffsets` (\ref TNL::Matrices::MultidiagonalMatrix::setDiagonalsOffsets) can be used. Note, however, that this method deletes all current matrix elements.

Another way of setting the matrix elements is by means of the method `setElement` (\ref TNL::Matrices::MutlidiagonalMatrix::setElement). It works the same way as with other matrix types as we can see in the follofwing example:

\includelineno MultidiagonalMatrixExample_setElement.cpp

This examples shows that the method `setElement` can be used both on the host (CPU) (line 17) as well as in the GPU kernels (lines 23-27). Here we use shared pointer (\ref TNL::Pointers::SharedPointer) (line 15) to pass the multidiagonal matrix to lambda function `f` (lines 22-28) which may run on GPU. In this case we have to synchronize to share pointer explicitly by calling the function \ref TNL::Pointers::synchronizeSmartPointersOnDevice. To avoid this inconvenience the same can be achieved with the multidiagonal matrix view:

\includelineno MultidiagonalMatrixViewExample_setElement.cpp

In this example, we fetch the matrix view (line 16) immediately after creating the matrix itself (line 15). Note that the matrix view can be obtained from the matrix at any time while the shared pointer only at the time of the matrix creation. On the other hand, if the original matrix is changed, all matrix views become invalid which is not true for the shared pointers. So it is better to fetch the matrix view immediately before we use it to avoid the sitaution that you would use invalid matrix view. The method `setElement` (\ref TNL::Matrices::MutlidiagonalMatrixView::setElement) can be used on both host (CPU) (line 19) and the device (lines 25-29) if the lambda function `f` (lines 24-30) runs in GPU kernel. The result of both examles looks the same:

\includelineno MultidiagonalMatrixViewExample_setElement.out

Another way for setting the matrix elements is by means of the multidiagonal matrix row:

\includelineno MultidiagonalMatrixViewExample_getRow.cpp

Here we use the matrix view again (line 19) and in the lambda function `f` which serves for the matrix elements setting, we fetch the matrix row just at the beginning (line 22). Next we use the method `setElement` (\ref TNL::Matrices::MultidiagonalMatrixRow::setElement) which accepts two parameters. The first is the local index of the matrix element which in case of the multidiagonal matrix agrees with index of the subdiagonal as demonstrated on this figure which shows just the matrix we are creating in this example (the subdiagonal indexes are depicted in blue color):

\f[
\begin{array}{c}
\color{blue}{0} \\
\hline
* \\
  \\
  \\
  \\
~
\end{array}
\left(
\begin{array}{ccccc}
 \color{blue}{1} &  \color{blue}{2} &    &    &    \\
 \hline
2  & -1 &    &    &    \\
-1 &  2 & -1 &    &    \\
   & -1 &  2 & -1 &    \\
   &    & -1 &  2 & -1 \\ 
   &    &    & -1 &  2
\end{array}
\right)
\f]

The second parameter of the method `setElement` is the new matrix elements value. An adventage of this method is that it can acces  the matrix elements faster. The output of this example looks as follows:

\includelineno MultidiagonalMatrixViewExample_getRow.out

Similar and even a bit simpler way of setting the matrix elements is offered by the method `forRows` (\ref TNL::Matrices::MultidiagonalMatrix::forRows, \ref TNL::Matrices::MultidiagonalMatrixView::forRows) as demonstrated in the following example:

\includelineno MultidiagonalMatrixViewExample_forRows.cpp

In this case, we need to provide a lambda function `f` (lines 27-43) which is called for each matrix row just by the method `forRows` (line 44). The lambda function `f` provides the following parameters

* `rowIdx` is an index iof the matrix row.
* `localIdx` is in index of the matrix subdiagonal.
* `columnIdx` is a column index of the matrix element.
* `value` is a reference to the matrix element value. It can be used even for changing the value.
* `compute` is a reference to boolean. If it is set to false, the iteration over the matrix row can be stopped.

In this example, the matrix element value depends only on the subdiagonal index `localIdx` as we can see on the line 42. The result looks as follows:

\includelineno MultidiagonalMatrixExample_forRows.out

### Flexible reduction in matrix rows

The flexible parallel reduction in rows for multidiagonal matrices works the same way as for other matrix types. It consits of three lambda functions:

1. `fetch` reads and preproces data entering the flexible parallel reduction.
2. `reduce` performs the reduction operation.
3. `keep` stores the results from each matrix row.

See the following example:

\includelineno MultidiagonalMatrixExample_rowsReduction.cpp

On the lines 10-29, we first create the following matrix

\f[
\left(
\begin{array}{ccccc}
1  &   &   &   &  \\
2  & 1 &   &   &  \\
3  & 2 & 1 &   &  \\
   & 3 & 2 & 1 &  \\
   &   & 3 & 2 & 1
\end{array}
\right)
\f]

and we aim to compute maximal value in each row. We first create vector `rowMax` into which we will store the results and fetch it view `rowMaxView` (line 39). Next we prepare necessary lambda functions:

* `fetch` (lines 44-46) is responsible for reading the matrix element value which is stored in the constant reference `value` and for returning its absolute value. The other parameters `rowIdx` and `columnIdx` correspond to row and column indexes respectively and they are omitted in our example.
* `reduce` (lines 51-53) returns maximum value of the two input values `a` and `b`.
* `keep` (line 58-60) stores the input `value` at the corresponding position, given by the row index `rowIdx`, in the ouput vector view `rowMaxView`.

Finaly we call the method `rowsReduction` (\ref TNL::Matrices::MultidiagonalMatrix::rowsReduction) with parameters telling the interval of rows to be processed (the first and second parameter), the lambda functions `fetch`, `reduce` and `keep`, and the idempotent element for the reduction operation which is the lowest number of given type (\ref std::numeric_limits< double >::lowest ). The result looks as follows:

\includelineno MultidiagonalMatrixExample_rowsReduction.out

### Multidiagonal matrix-vector product

Similar to matrix types, matrix-vector multiplication is represented by the method `vectorProduct` (\ref TNL::Matrices::MultidiagonalMatrix::vectorProduct). It is templated method with two template parameters `InVector` and `OutVector` telling the types of the input and output vector respectively. Usually one will substitute some of \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView for these types. The method computes the following formula

```
outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector
```

and it accepts the following parameters:

* `inVector` is the input vector having the same number of elements as the number of matrix columns.
* `outVector` is the output vector having the same number of elements as the number of matrix rows.
* `matrixMultiplicator` is a number by which the result of matrix-vector product is multiplied. 
* `outVectorMultiplicator` is a number by which the output vector is multiplied before it is added to the result of matrix-vector product.
* `begin` is an index of the first matrix row that is involved in the multiplication. It is zero be default.
* `end` is an index indicating the last matrix row that is involved in the multiplication which is `end - 1`. It is the number of matrix rows.

Note that the output vector dimension must be the same as the number of matrix rows no matter how we set `begin` and `end` parameters. These parameters just say that some matrix rows and the output vector elements are omitted.

### Multidiagonal matrix IO

The multidiagonal matrix can be saved to a file using a method `save` (\ref TNL::Matrices::MultiidiagonalMatrix::save) and restored with a method `load` (\ref TNL::Matrices::MultidiagonalMatrix::load). For printing the matrix, there is a method `print` (\ref TNL::Matrices::MultidiagonalMatrix::print) can be used.

### Multidiagonal matrix view <a name="multidiagonal_matrix_view"></a>

Multidiagonal matrix also offers its view for easier use with lambda functions. It is represented by a templated class \ref TNL::Matrices::MultidiagonalMatrixView with the following template parameters:

* `Real` is a type of matrix elements. 
* `Device` is a device on which the matrix is allocated. This can be \ref TNL::Devices::Host or \ref TNL::Devices::Cuda.
* `Index` is a type for indexing the matrix elements and also row and column indexes.
* `Organization` tells the ordering of matrix elements in memory. It is either RowMajorOrder or ColumnMajorOrder.

The first main reason for using the matrix view is its ability to be captured by lambda functions since the copy constructor makes only shallow copy. We can demonstrate it on the example showing the method `setElement` (\ref TNL::Matrices::MultidiagonalMatrix::setElement). The code looks as follows:

\includelineno MultidiagonalMatrixViewExample_setElement.cpp

The matrix view is obtained by the method `getView` (\ref TNL::Matrices::MultidiagonalMatrix::getView) on the line 13. We firsrt show, that the view can be used the same way as common matrix (lines 14 and 15) but it can be used the same way even in lambda functions as we can see on the lines 20-26. Compare it with the same example using shared pointer instead of the matrix view:

\includelineno MultidiagonalMatrixExample_setElement.cpp

The main disadventages are:

1. The shared pointer must be created together with the matrix (line 14) and there is no way to get it later. The matrix view can be obtained from any matrix at any time.
2. We have to synchronize shared pointers explicitly by calling the function \ref TNL::Pointers::synchronizeSmartPointersOnDevice (line 34).

So for the sake of using a matrix in lambda functions, the matrix view is better tool. The result of both examples looks as:

\includelineno MultidiagonalMatrixExample_setElement.out

As we mentioned already, the multidiagonal matrix view offers almost all methods which the multidiagonal matrix does. So it can be easily used at almost any situation the same way as the multidiagonal matrix itself.

TODO: Move to explanation of the matrix view to introduction.

## Lambda matrices <a name="lambda_matrices"></a>

Lambda matrix (\ref TNL::Matrices::LambdaMatrix) is a special type of matrix which could be also called *** matrix-free matrix ***. Its elements are not stored in memory explicitlely but they are evaluated on-the-fly by means of user defined lambda functions. If the matrix elements can be expressed by computationaly not expansive formula, we can significantly reduce the memory consumptions which can be appriciated especially on GPU. Since the memory accesses are quite expensive even on CPU, we can get, at the end, even much faster code.

The lambda matrix (\ref TNL::Matrices::LambdaMatrix) is a templated class with the following template parameters:

* `MatrixElementsLambda` is a lambda function which evaluates the matrix elements values and column indexes.
* `CompressedRowLengthsLambda` is a lambda function telling how many nonzero elements are there in given matrix row.
* `Real` is a of matrix elements values.
* `Device` is a device on which the lambda functions mentioned above will be evaluated.
* `Index` is a type to be used for indexing.

The lambda function `MatrixElementsLambda` is supposed to have the following declaration:

```
matrixElements( Index rows, 
                Index columns,
                Index row,
                Index localIdx,
                Index& columnIdx,
                Real& value )
```
where the particular parameterts have the following meaning:

* `rows` tells the number of matrix rows.
* `columns` tells the number of matrix columns.
* `rowIdx` is index of the matrix row in which we are supposed to evaluate the matrix element.
* `localIdx` is a rank of the nonzero matrix element.
* `columnIdx` is a reference on variable where we are supposed to store the matrix element column index.
* `value` is a reference on variable where we are supposed to store the matrix element value.

The lambda function `CompressedRowLengthsLambda` is supposed to look like this:

```
rowLengths( Index rows, 
            Index columns,
            Index row ) -> Index
```

where the parameters can be described as follows:

* `rows` tells the number of matrix rows.
* `columns` tells the number of matrix columns.
* `rowIdx` is index of the matrix row for which we are supposed to evaluate the number of nonzero matrix elements.

The lambda function is supposed to return just the number of the nonzero matrix elements in given matrix row.

### Lambda matrix inititation

See the following example which demonstrates how to create the lambda matrix:

\includelineno LambdaMatrixExample_Constructor.cpp

Here we create two simple diagonal matrices. Therefore thay share the same lambda function `rowLengths` telling the the number of nonzero matrix elements in particular matrix rows which is always one (line 9). The first matrix, defined by the lambda function `matrixElements1`, is identity matrix and so its each diagonal element equals one. We set the matrix element value to `1.0` (line 12) and the column index equals the row index (line 15). The second matrix, defined by the lambda function `matrixElements2`, is also diagonal but not the identity matrix. The values of the diagonal elements equal to row index (line 16).

With the same lambda functions we can define matrices with different dimensions. In this example, we set the matrix size to five (line 19). It can be quite difficult to express the lambda matrix type because it depends on the types of the lambda functions. To make this easier, one may use a lambda-matrix factory (\ref TNL::Matrices::LambdaMatrixFactory). Using `decltype` one can deduce even the matrix type (line 24) followed by calling lambda matrix constructor with matrix dimensions and instances of the lambda functions (line 25). Or one can just simply employ the keyword `auto` (line 30) followed by setting the matrix dimensins (line 31).

The result looks as follows:

\includelineno LambdaMatrixExample_Constructor.out

Of course, the lambda matrix has the same interface as other matrix types. The following example demonstrates the use of the method `forRows` to copy the lambda matrix into the dense matrix:

\includelineno LambdaMatrixExample_forRows.cpp

Here, we treat the lambda matrix as if it was dense matrix. The lambda function `rowLengths` returns the number of the nonzero elements equal to the number of matrix columns (line 13). However, the lambda function `matrixElements` (lines 14-17), sets nozero values only to lower triangular part of the matrix. The elements in the upper part are equal to zero (line 16). Next we create an instance of the lambda matrix with help of the lambda matrix factory (\ref TNL::Matrices::LambdaMatrixFactory) (lines 19-20) and an instance of the dense matrix (\ref TNL::Matrices::DenseMatrix) (lines 22-23). 

Next we call the lambda function `f` by the method `forRows` (\ref TNL::Matrices::LambdaMatrix::forRows) to set the matrix elements of the dense matrix `denseMatrix` (line 26) via the dense matrix view (`denseView`) (\ref TNL::Matrices::DenseMatrixView). Note, that in the lambda function `f` we get the matrix element value already evaluated in the variable `value` as we are used to from other matrix types. So in fact, the same lambda function `f` woudl do the same job even for sparse matrix or any other. Also note, that in this case we iterate even over all zero matrix elements because the lambda function `rowLengths` (line 13) tells so. The result looks as follows:

\includelineno LambdaMatrixExample_forRows.out

### Flexible reduction in matrix rows

The reduction of matrix rows is available for the lambda matrices as well. See the follogin example:

\includelineno LambdaMatrixExample_rowsReduction.cpp

On the lines 14-21, we create the same lower trianguilar lambda matrix as in the previous example. As we did it in similar examples for other matrix types, we want to compute maximal absolute value of matrix elements in each row. For this purpose we define well known lambda functions:

* `fetch` takes the value of the lambda matrix element and returns its absolute value.
* `reduce` computes maximum value of two input variables.
* `keep` stores the results into output vector `rowMax`.

Note that the interface of the lambda functions is the same as for other matrix types. The result looks as follows:

\includelineno LambdaMatrixExample_rowsReduction.out

### Lambda matrix-vector product

The matrix-vector multiplication is represented by the method `vectorProduct` (\ref TNL::Matrices::LambdaMatrix::vectorProduct). It is templated method with two template parameters `InVector` and `OutVector` telling the types of the input and output vector respectively. Usually one will substitute some of \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView for these types. The method computes the following formula

```
outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector
```

and it accepts the following parameters:

* `inVector` is the input vector having the same number of elements as the number of matrix columns.
* `outVector` is the output vector having the same number of elements as the number of matrix rows.
* `matrixMultiplicator` is a number by which the result of matrix-vector product is multiplied. 
* `outVectorMultiplicator` is a number by which the output vector is multiplied before it is added to the result of matrix-vector product.
* `begin` is an index of the first matrix row that is involved in the multiplication. It is zero be default.
* `end` is an index indicating the last matrix row that is involved in the multiplication which is `end - 1`. It is the number of matrix rows.

Note that the output vector dimension must be the same as the number of matrix rows no matter how we set `begin` and `end` parameters. These parameters just say that some matrix rows and the output vector elements are omitted.

### Lambda matrix IO

The lambda matrix, can be printed by the means of the method `print` (\ref TNL::Matrices::LambdaMatrix::print). The lambda matrix do not offer the methods `save` and `load` since it does not manage any data. Of course, the lambda function evaluating the matrix elements can use any supporting data containers but it is up these containers to manage the IO operations.