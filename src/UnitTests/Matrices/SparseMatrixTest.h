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
using E_host = TNL::Matrices::Ellpack< int, TNL::Devices::Host, int >;
using E_cuda = TNL::Matrices::Ellpack< int, TNL::Devices::Cuda, int >;
using SE_host = TNL::Matrices::SlicedEllpack< int, TNL::Devices::Host, int, 2 >;
using SE_cuda = TNL::Matrices::SlicedEllpack< int, TNL::Devices::Cuda, int, 2 >;

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
//TODO Tests go in here

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

