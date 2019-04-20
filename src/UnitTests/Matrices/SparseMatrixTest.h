/***************************************************************************
                          SparseMatrixTest.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Matrices/CSR.h>

#include "SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

using CSR_host_float = TNL::Matrices::CSR< float, TNL::Devices::Host, int >;
using CSR_host_int = TNL::Matrices::CSR< int, TNL::Devices::Host, int >;

using CSR_cuda_float = TNL::Matrices::CSR< float, TNL::Devices::Cuda, int >;
using CSR_cuda_int = TNL::Matrices::CSR< int, TNL::Devices::Cuda, int >;

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
}
#endif

#endif

#include "../main.h"
