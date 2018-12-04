/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// TODO
/*
 * getType()                        ::HOW?  How to test this for each format? edit string how?
 *      Found the mistake for Cuda instead of Devices::Cuda. Incorrect String in src/TNL/Devices/Cuda.cpp
 *      MISSING: indexType is missing in CSR_impl.h
 * getTypeVirtual()                 ::TEST? This just calls getType().
 * getSerializationType()           ::TEST? This just calls HostType::getType().
 * getSerializationTypeVirtual()    ::TEST? This just calls getSerializationType().
 * setDimensions()                      ::DONE
 * setCompressedRowLengths()            ::DONE
 * getRowLength()                   ::USED! In test_SetCompressedRowLengths() to verify the test itself.
 * getRowLengthFast()               ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setLike()                            ::DONE
 * reset()                              ::DONE
 * setElementFast()                 ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setElement()                         ::DONE
 * addElementFast()                 ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * addElement()                         ::DONE
 * setRowFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * setRow()                             ::DONE
 *      MISTAKE!!! In SlicedEllpack: addElement(), line 263, "column <= this->rows" shouldn't it be: "column <= this->columns", otherwise test_SetRow causes the assertion to fail.
 * addRowFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * addRow()                         ::NOT IMPLEMENTED! This calls addRowFast() which isn't implemented. Implement? Is it supposed to add an extra row to the matrix or add elements of a row to another row in the matrix?
 * getElementFast()                 ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * getElement()                     ::USED! In test_SetElement(), test_AddElement() and test_setRow() to verify the test itself.
 * getRowFast()                     ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * MatrixRow getRow()               ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * ConstMatrixRow getRow()          ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * rowVectorProduct()               ::TEST? How to test __cuda_callable__? ONLY TEST ON CPU FOR NOW
 * vectorProduct()                      ::DONE
 *      This used to throw illegal memory access, but instead of using ints for vectors, using Types, helped.
 * addMatrix()                      ::NOT IMPLEMENTED!
 * getTransposition()               ::NOT IMPLMENETED!
 * performSORIteration()            ::HOW? Throws segmentation fault CUDA.
 * operator=()                      ::HOW? What is this supposed to enable? Overloading operators?
 * save( File& file)                ::USED! In save( String& fileName )
 * load( File& file )               ::USED! In load( String& fileName )
 * save( String& fileName )             ::DONE
 * load( String& fileName )             ::DONE
 * print()                              ::DONE
 * setCudaKernelType()              ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * getCudaKernelType()              ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * setCudaWarpSize()                ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * getCudaWarpSize()                ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * setHybridModeSplit()             ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * getHybridModeSplit()             ::NOT SUPPOSED TO TEST! via notes from 1.11.2018 supervisor meeting.
 * spmvCudaVectorized()             ::TEST? How to test __device__?
 * vectorProductCuda()              ::TEST? How to test __device__?
 */

// GENERAL TODO
/*
 * For every function, EXPECT_EQ needs to be done, even for zeros in matrices.
 * Figure out __cuda_callable_. When trying to call __cuda_callable__ functions
 *      a segmentation fault (core dumped) is thrown.
 *  ==>__cuda_callable__ works only for CPU at the moment. (for loops vs thread kernel assignment)
 */

#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>

#include <TNL/Matrices/ChunkedEllpack.h>
#include <TNL/Matrices/AdEllpack.h>
#include <TNL/Matrices/BiEllpack.h>

#include <UnitTests/Matrices/SparseMatrixTest_impl.h>
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

using CSR_host_float = TNL::Matrices::CSR< float, TNL::Devices::Host, int >;
using CSR_host_int = TNL::Matrices::CSR< int, TNL::Devices::Host, int >;

using CSR_cuda_float = TNL::Matrices::CSR< float, TNL::Devices::Cuda, int >;
using CSR_cuda_int = TNL::Matrices::CSR< int, TNL::Devices::Cuda, int >;

#ifdef NOT_WORKING
// test fixture for typed tests
template< typename Matrix >
class AdEllpackMatrixTest : public ::testing::Test
{
protected:
   using AdEllpackMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using AdEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Host, short >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Host, int >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Host, long >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Host, long >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Host, long >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Host, long >,
#ifdef HAVE_CUDA
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::AdEllpack< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::AdEllpack< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::AdEllpack< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::AdEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_CASE( AdEllpackMatrixTest, AdEllpackMatrixTypes);

TYPED_TEST( AdEllpackMatrixTest, setDimensionsTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetDimensions< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setCompressedRowLengthsTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetCompressedRowLengths< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setLikeTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetLike< AdEllpackMatrixType, AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, resetTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_Reset< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setElementTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetElement< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, addElementTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_AddElement< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setRowTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetRow< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, vectorProductTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_VectorProduct< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, saveAndLoadTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SaveAndLoad< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, printTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_Print< AdEllpackMatrixType >();
}

// test fixture for typed tests
template< typename Matrix >
class BiEllpackMatrixTest : public ::testing::Test
{
protected:
   using BiEllpackMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using BiEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::BiEllpack< int,    TNL::Devices::Host, short >,
    TNL::Matrices::BiEllpack< long,   TNL::Devices::Host, short >,
    TNL::Matrices::BiEllpack< float,  TNL::Devices::Host, short >,
    TNL::Matrices::BiEllpack< double, TNL::Devices::Host, short >,
    TNL::Matrices::BiEllpack< int,    TNL::Devices::Host, int >,
    TNL::Matrices::BiEllpack< long,   TNL::Devices::Host, int >,
    TNL::Matrices::BiEllpack< float,  TNL::Devices::Host, int >,
    TNL::Matrices::BiEllpack< double, TNL::Devices::Host, int >,
    TNL::Matrices::BiEllpack< int,    TNL::Devices::Host, long >,
    TNL::Matrices::BiEllpack< long,   TNL::Devices::Host, long >,
    TNL::Matrices::BiEllpack< float,  TNL::Devices::Host, long >,
    TNL::Matrices::BiEllpack< double, TNL::Devices::Host, long >//,
//#ifdef HAVE_CUDA
//    TNL::Matrices::BiEllpack< int,    TNL::Devices::Cuda, short >,
//    TNL::Matrices::BiEllpack< long,   TNL::Devices::Cuda, short >,
//    TNL::Matrices::BiEllpack< float,  TNL::Devices::Cuda, short >,
//    TNL::Matrices::BiEllpack< double, TNL::Devices::Cuda, short >,
//    TNL::Matrices::BiEllpack< int,    TNL::Devices::Cuda, int >,
//    TNL::Matrices::BiEllpack< long,   TNL::Devices::Cuda, int >,
//    TNL::Matrices::BiEllpack< float,  TNL::Devices::Cuda, int >,
//    TNL::Matrices::BiEllpack< double, TNL::Devices::Cuda, int >,
//    TNL::Matrices::BiEllpack< int,    TNL::Devices::Cuda, long >,
//    TNL::Matrices::BiEllpack< long,   TNL::Devices::Cuda, long >,
//    TNL::Matrices::BiEllpack< float,  TNL::Devices::Cuda, long >,
//    TNL::Matrices::BiEllpack< double, TNL::Devices::Cuda, long >
//#endif
>;

TYPED_TEST_CASE( BiEllpackMatrixTest, BiEllpackMatrixTypes);

TYPED_TEST( BiEllpackMatrixTest, setDimensionsTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_SetDimensions< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, setCompressedRowLengthsTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_SetCompressedRowLengths< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, setLikeTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_SetLike< BiEllpackMatrixType, BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, resetTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_Reset< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, setElementTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_SetElement< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, addElementTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_AddElement< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, setRowTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_SetRow< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, vectorProductTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_VectorProduct< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, saveAndLoadTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_SaveAndLoad< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, printTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
    
    test_Print< BiEllpackMatrixType >();
}

#endif

// GTEST ::testing::Types<> has a limit of 38.

// test fixture for typed tests
template< typename Matrix >
class ChunkedEllpackMatrixTest : public ::testing::Test
{
protected:
   using ChunkedEllpackMatrixType = Matrix;
};

// columnIndexes of ChunkedEllpack appear to be broken, when printed, it prints out a bunch of 4s.
// rowPointers have interesting elements? 0 18 36 42 54 72 96 126 162 204 256 when rows = 10, cols = 11; rowLengths = 3 3 1 2 3 4 5 6 7 8
// and 0 52 103 154 205 256 when rows = 5, cols = 4; rowLengths = 3 3 3 3 3


// types for which MatrixTest is instantiated
using ChEllpackMatrixTypes = ::testing::Types
<
//    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Host, short >,
//    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Host, short >,
//    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Host, short >,
//    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Host, short >,
//    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Host, int >,
//    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Host, int >,
//    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Host, int >,
//    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Host, int >,
//    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Host, long >,
//    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Host, long >,
//    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Host, long >,
//    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Host, long >,
#ifdef HAVE_CUDA
//    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Cuda, short >,
//    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Cuda, short >,
//    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Cuda, short >,
//    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Cuda, short >,
//    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Cuda, int >,
//    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Cuda, int >,
//    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Cuda, int >,
//    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Cuda, int >,
//    TNL::Matrices::ChunkedEllpack< int,    TNL::Devices::Cuda, long >,
//    TNL::Matrices::ChunkedEllpack< long,   TNL::Devices::Cuda, long >,
//    TNL::Matrices::ChunkedEllpack< float,  TNL::Devices::Cuda, long >,
//    TNL::Matrices::ChunkedEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_CASE( ChunkedEllpackMatrixTest, ChEllpackMatrixTypes);

TYPED_TEST( ChunkedEllpackMatrixTest, setDimensionsTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetDimensions< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setCompressedRowLengthsTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetCompressedRowLengths< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setLikeTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetLike< ChunkedEllpackMatrixType, ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, resetTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_Reset< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setElementTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetElement< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, addElementTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_AddElement< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setRowTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetRow< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, vectorProductTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_VectorProduct< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, saveAndLoadTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SaveAndLoad< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, printTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_Print< ChunkedEllpackMatrixType >();
}

// test fixture for typed tests
template< typename Matrix >
class CSRMatrixTest : public ::testing::Test
{
protected:
   using CSRMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using CSRMatrixTypes = ::testing::Types
<
//    TNL::Matrices::CSR< int,    TNL::Devices::Host, short >,
//    TNL::Matrices::CSR< long,   TNL::Devices::Host, short >,
//    TNL::Matrices::CSR< float,  TNL::Devices::Host, short >,
//    TNL::Matrices::CSR< double, TNL::Devices::Host, short >,
//    TNL::Matrices::CSR< int,    TNL::Devices::Host, int >,
//    TNL::Matrices::CSR< long,   TNL::Devices::Host, int >,
//    TNL::Matrices::CSR< float,  TNL::Devices::Host, int >,
//    TNL::Matrices::CSR< double, TNL::Devices::Host, int >,
//    TNL::Matrices::CSR< int,    TNL::Devices::Host, long >,
//    TNL::Matrices::CSR< long,   TNL::Devices::Host, long >,
//    TNL::Matrices::CSR< float,  TNL::Devices::Host, long >,
    TNL::Matrices::CSR< double, TNL::Devices::Host, long >,
#ifdef HAVE_CUDA
//    TNL::Matrices::CSR< int,    TNL::Devices::Cuda, short >,
//    TNL::Matrices::CSR< long,   TNL::Devices::Cuda, short >,
//    TNL::Matrices::CSR< float,  TNL::Devices::Cuda, short >,
//    TNL::Matrices::CSR< double, TNL::Devices::Cuda, short >,
//    TNL::Matrices::CSR< int,    TNL::Devices::Cuda, int >,
//    TNL::Matrices::CSR< long,   TNL::Devices::Cuda, int >,
//    TNL::Matrices::CSR< float,  TNL::Devices::Cuda, int >,
//    TNL::Matrices::CSR< double, TNL::Devices::Cuda, int >,
//    TNL::Matrices::CSR< int,    TNL::Devices::Cuda, long >,
//    TNL::Matrices::CSR< long,   TNL::Devices::Cuda, long >,
//    TNL::Matrices::CSR< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::CSR< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_CASE( CSRMatrixTest, CSRMatrixTypes);

TYPED_TEST( CSRMatrixTest, setDimensionsTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_SetDimensions< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setCompressedRowLengthsTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_SetCompressedRowLengths< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setLikeTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_SetLike< CSRMatrixType, CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, resetTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_Reset< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setElementTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_SetElement< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, addElementTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_AddElement< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, setRowTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_SetRow< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, vectorProductTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_VectorProduct< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, saveAndLoadTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_SaveAndLoad< CSRMatrixType >();
}

TYPED_TEST( CSRMatrixTest, printTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;
    
    test_Print< CSRMatrixType >();
}

// test fixture for typed tests
template< typename Matrix >
class EllpackMatrixTest : public ::testing::Test
{
protected:
   using EllpackMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using EllpackMatrixTypes = ::testing::Types
<
//    TNL::Matrices::Ellpack< int,    TNL::Devices::Host, short >,
//    TNL::Matrices::Ellpack< long,   TNL::Devices::Host, short >,
//    TNL::Matrices::Ellpack< float,  TNL::Devices::Host, short >,
//    TNL::Matrices::Ellpack< double, TNL::Devices::Host, short >,
//    TNL::Matrices::Ellpack< int,    TNL::Devices::Host, int >,
//    TNL::Matrices::Ellpack< long,   TNL::Devices::Host, int >,
//    TNL::Matrices::Ellpack< float,  TNL::Devices::Host, int >,
//    TNL::Matrices::Ellpack< double, TNL::Devices::Host, int >,
//    TNL::Matrices::Ellpack< int,    TNL::Devices::Host, long >,
//    TNL::Matrices::Ellpack< long,   TNL::Devices::Host, long >,
//    TNL::Matrices::Ellpack< float,  TNL::Devices::Host, long >,
//    TNL::Matrices::Ellpack< double, TNL::Devices::Host, long >,
#ifdef HAVE_CUDA
//    TNL::Matrices::Ellpack< int,    TNL::Devices::Cuda, short >,
//    TNL::Matrices::Ellpack< long,   TNL::Devices::Cuda, short >,
//    TNL::Matrices::Ellpack< float,  TNL::Devices::Cuda, short >,
//    TNL::Matrices::Ellpack< double, TNL::Devices::Cuda, short >,
//    TNL::Matrices::Ellpack< int,    TNL::Devices::Cuda, int >,
//    TNL::Matrices::Ellpack< long,   TNL::Devices::Cuda, int >,
//    TNL::Matrices::Ellpack< float,  TNL::Devices::Cuda, int >,
//    TNL::Matrices::Ellpack< double, TNL::Devices::Cuda, int >,
//    TNL::Matrices::Ellpack< int,    TNL::Devices::Cuda, long >,
//    TNL::Matrices::Ellpack< long,   TNL::Devices::Cuda, long >,
//    TNL::Matrices::Ellpack< float,  TNL::Devices::Cuda, long >,
//    TNL::Matrices::Ellpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_CASE( EllpackMatrixTest, EllpackMatrixTypes );

TYPED_TEST( EllpackMatrixTest, setDimensionsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_SetDimensions< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setCompressedRowLengthsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_SetCompressedRowLengths< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setLikeTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_SetLike< EllpackMatrixType, EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, resetTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_Reset< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_SetElement< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, addElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_AddElement< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setRowTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_SetRow< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, vectorProductTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_VectorProduct< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, saveAndLoadTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_SaveAndLoad< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, printTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
    
    test_Print< EllpackMatrixType >();
}

// test fixture for typed tests
template< typename Matrix >
class SlicedEllpackMatrixTest : public ::testing::Test
{
protected:
   using SlicedEllpackMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
//    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Host, short >,
//    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Host, short >,
//    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Host, short >,
//    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, short >,
//    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Host, int >,
//    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Host, int >,
//    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Host, int >,
//    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, int >,
//    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Host, long >,
//    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Host, long >,
//    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Host, long >,
//    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, long >,
#ifdef HAVE_CUDA
//    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Cuda, short >,
//    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Cuda, short >,
//    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Cuda, short >,
//    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, short >,
//    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Cuda, int >,
//    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Cuda, int >,
//    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Cuda, int >,
//    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, int >,
//    TNL::Matrices::SlicedEllpack< int,    TNL::Devices::Cuda, long >,
//    TNL::Matrices::SlicedEllpack< long,   TNL::Devices::Cuda, long >,
//    TNL::Matrices::SlicedEllpack< float,  TNL::Devices::Cuda, long >,
//    TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_CASE( SlicedEllpackMatrixTest, SlicedEllpackMatrixTypes );

TYPED_TEST( SlicedEllpackMatrixTest, setDimensionsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_SetDimensions< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setCompressedRowLengthsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_SetCompressedRowLengths< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setLikeTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_SetLike< SlicedEllpackMatrixType, SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, resetTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_Reset< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_SetElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, addElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_AddElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setRowTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_SetRow< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, vectorProductTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_VectorProduct< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, saveAndLoadTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_SaveAndLoad< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, printTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
    
    test_Print< SlicedEllpackMatrixType >();
}

//// test_getType is not general enough yet. DO NOT TEST IT YET.

//TEST( SparseMatrixTest, CSR_GetTypeTest_Host )
//{
//    host_test_GetType< CSR_host_float, CSR_host_int >();
//}
//
//#ifdef HAVE_CUDA
//TEST( SparseMatrixTest, CSR_GetTypeTest_Cuda )
//{
//    cuda_test_GetType< CSR_cuda_float, CSR_cuda_int >();
//}
//#endif

TEST( SparseMatrixTest, CSR_perforSORIterationTest_Host )
{
    test_PerformSORIteration< CSR_host_float >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_perforSORIterationTest_Cuda )
{
//    test_PerformSORIteration< CSR_cuda_float >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched, this test throws the following message: \n";
    std::cout << "      [1]    16958 segmentation fault (core dumped)  ./SparseMatrixTest-dbg\n\n";
    std::cout << "\n THIS IS NOT IMPLEMENTED FOR CUDA YET!!\n\n";
}
#endif

#endif

#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}