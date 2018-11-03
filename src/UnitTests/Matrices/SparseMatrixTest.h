/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>

using CSR_host = TNL::Matrices::CSR< int, TNL::Devices::Host, int >;
using CSR_cuda = TNL::Matrices::CSR< int, TNL::Devices::Cuda, int >;

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

template< typename Matrix >
void testGetType()
{
    Matrix<float, TNL::Devices::Cuda, int> floatCudaMatrix;
//    using CSR_host_getType = TNL::Matrices::CSR< float, TNL::Devices::Host, int>
    Matrix<float, TNL::Devices::Host, int> floatHostMatrix;
//    using CSR_cuda_getType = TNL::Matrices::CSR< float, TNL::Devices::Cuda, int>
    EXPECT_EQ( floatCudaMatrix.getType(), "Matrices::CSR< float, Cuda >");
}

TEST( SparseMatrixTest, GetTypeTest )
{
   testGetType< CSR_host >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, GetTypeTest )
{
   testGetType< CSR_cuda >();
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

