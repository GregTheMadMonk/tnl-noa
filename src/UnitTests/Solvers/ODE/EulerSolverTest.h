#pragma once

#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Algorithms/ParallelFor.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_GTEST

// test fixture for typed tests
template< typename DofContainer >
class ODENumericSolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

template< typename DofContainer >
class ODEStaticSolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

template< typename DofContainer >
class ODESolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

// types for which DofContainerTest is instantiated
using DofNumericTypes = ::testing::Types<
   float,
   double
>;

// types for which DofContainerTest is instantiated
using DofStaticVectorTypes = ::testing::Types<
   StaticVector< 1, float >,
   StaticVector< 2, float >,
   StaticVector< 3, float >,
   StaticVector< 1, double >,
   StaticVector< 2, double >,
   StaticVector< 3, double >
>;

// types for which DofContainerTest is instantiated
using DofVectorTypes = ::testing::Types<
#ifndef HAVE_CUDA
   // we can't test all types because the argument list would be too long...
   Vector< float,  Devices::Sequential, int >,
   Vector< double, Devices::Sequential, int >,
   Vector< float,  Devices::Sequential, long >,
   Vector< double, Devices::Sequential, long >
#endif
#ifdef HAVE_CUDA
  ,Vector< float,  Devices::Cuda, int >,
   Vector< double, Devices::Cuda, int >,
   Vector< float,  Devices::Cuda, long >,
   Vector< double, Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ODENumericSolverTest, DofNumericTypes );

TYPED_TEST_SUITE( ODEStaticSolverTest, DofStaticVectorTypes );

TYPED_TEST_SUITE( ODESolverTest, DofVectorTypes );

TYPED_TEST( ODENumericSolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = TNL::Solvers::ODE::Euler< DofContainerType >;
   using Real = DofContainerType;

   const Real final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 0.0 );
   solver.solve( u, [] ( const Real& time, const Real& tau, const auto& u, auto& fu ) {
      fu = time;
   } );

   Real exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::abs( u - exact_solution ), ( Real ) 0.0, 0.1 );
}

TYPED_TEST( ODEStaticSolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = TNL::Solvers::ODE::Euler< DofContainerType >;
   using Real = typename DofContainerType::RealType;

   const Real final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 0.0 );
   solver.solve( u, [] ( const Real& time, const Real& tau, const auto& u, auto& fu ) {
      fu = time;
   } );

   Real exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), ( Real ) 0.0, 0.1 );
}

TYPED_TEST( ODESolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = TNL::Solvers::ODE::Euler< DofContainerType >;
   using Real = typename DofContainerType::RealType;

   const Real final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 5, 0.0 );
   solver.solve( u, [] ( const Real& time, const Real& tau, auto u, auto fu ) {
      fu = time;
   } );

   Real exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), ( Real ) 0.0, 0.1 );
   //std::cout << u << std::endl;
}


#endif

#include "../../main.h"
