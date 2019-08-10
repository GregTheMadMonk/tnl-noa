/***************************************************************************
                          MpiCommunicatorTest.h  -  description
                             -------------------
    begin                : Jul 10, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_GTEST

#include "gtest/gtest.h"
#include <TNL/Communicators/MpiCommunicator.h>

using namespace TNL;
using namespace TNL::Communicators;

// test fixture for typed tests
template< typename Real >
class MpiCommunicatorTest : public ::testing::Test
{
   protected:
      using RealType = Real;
      using CommunicatorType = MpiCommunicator;
};

// types for which MpiCommunicatorTest is instantiated
using MpiCommunicatorTypes = ::testing::Types<
   short,
   int,
   long,
   float,
   double
>;

TYPED_TEST_SUITE( MpiCommunicatorTest, MpiCommunicatorTypes );

TYPED_TEST( MpiCommunicatorTest, allReduce )
{
   using RealType = typename TestFixture::RealType;
   using CommunicatorType = typename TestFixture::CommunicatorType;
   RealType a = CommunicatorType::GetRank();
   RealType b = 0;
   CommunicatorType::Allreduce( &a, &b, 1, MPI_MAX, MPI_COMM_WORLD );
   EXPECT_EQ( b, CommunicatorType::GetSize() - 1  );
}

#endif // HAVE_GTEST

#include "../main_mpi.h"