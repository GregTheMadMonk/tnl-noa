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

using CSR_host_float = TNL::Matrices::CSR< float, TNL::Devices::Host, int >;
using CSR_host_int = TNL::Matrices::CSR< int, TNL::Devices::Host, int >;

using CSR_cuda_float = TNL::Matrices::CSR< float, TNL::Devices::Cuda, int >;
using CSR_cuda_int = TNL::Matrices::CSR< int, TNL::Devices::Cuda, int >;

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
/*

template< typename MatrixHostFloat, typename MatrixHostInt, typename MatrixCudaFloat, typename MatrixCudaInt >
void testGetType()
{
    MatrixHostFloat mtrxHostFloat;
    MatrixHostInt mtrxHostInt;
    MatrixCudaFloat mtrxCudaFloat;
    MatrixCudaInt mtrxCudaInt;
    
    //string str = "Matrices::CSR< float, Devices::Host >";
    
    EXPECT_STREQ( mtrxHostFloat.getType(), String("Matrices::CSR< float, Devices::Host >") );
    EXPECT_STREQ( mtrxHostInt.getType(), String("Matrices::CSR< int, Devices::Host >") );
    EXPECT_STREQ( mtrxCudaFloat.getType(), "Matrices::CSR< float, Cuda >" );
    EXPECT_STREQ( mtrxCudaInt.getType(), "Matrices::CSR< int, Cuda >" );
    
}

TEST( SparseMatrixTest, CSR_GetTypeTest )
{
   testGetType< CSR_host_float, CSR_host_int, CSR_cuda_float, CSR_cuda_int >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, GetTypeTestCuda )
{
   testGetType< CSR_host_float, CSR_host_int, CSR_cuda_float, CSR_cuda_int >();
}
#endif
*/
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

