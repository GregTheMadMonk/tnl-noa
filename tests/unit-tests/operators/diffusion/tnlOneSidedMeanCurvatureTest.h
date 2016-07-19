/***************************************************************************
                          tnlOneSidedMeanCurvatureTest.h  -  description
                             -------------------
    begin                : Feb 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMEANCURVATURETEST_H
#define	TNLMEANCURVATURETEST_H

#include <TNL/operators/diffusion/tnlOneSidedMeanCurvature.h>
#include <TNL/operators/diffusion/tnlExactLinearDiffusion.h>
#include <TNL/mesh/tnlGrid.h>
#include "../tnlPDEOperatorEocUnitTest.h"
#include "../tnlPDEOperatorEocTest.h"
#include "../../tnlUnitTestStarter.h"

using namespace TNL;

template< typename ApproximateOperator,
          typename TestFunction,
          bool write = false,
          bool verbose = false >
class tnlOneSidedMeanCurvatureTest
   : public tnlPDEOperatorEocTest< ApproximateOperator, TestFunction >
{
   public:
 
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename ApproximateOperatorType::ExactOperatorType ExactOperatorType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::IndexType IndexType;
 
      const IndexType coarseMeshSize[ 3 ] = { 128, 256, 64 };
 
      const RealType  eoc[ 3 ] =       { 2.0,  2.0,  2.0 };
      const RealType  tolerance[ 3 ] = { 0.05, 0.05, 0.05 };
 
      tnlOneSidedMeanCurvatureTest(){};
 
      static tnlString getType()
      {
         return tnlString( "tnlOneSidedMeanCurvatureTest< " ) +
                ApproximateOperator::getType() + ", " +
                TestFunction::getType() + " >";
      }
 
      void setupTest()
      {
         this->setupFunction();
      }
 
      void getApproximationError( const IndexType meshSize,
                                  RealType errors[ 3 ] )
      {
         this->setupMesh( meshSize );
         ApproximateOperator approximateOperator( this->mesh );
         ExactOperatorType exactOperator;
         approximateOperator.setRegularizationEpsilon( 1.0 );
         exactOperator.setRegularizationEpsilon( 1.0 );
         this->performTest( approximateOperator,
                            exactOperator,
                            errors,
                            write,
                            verbose );

      }
 
      void runUnitTest()
      {
         RealType coarseErrors[ 3 ], fineErrors[ 3 ];
         this->getApproximationError( coarseMeshSize[ MeshType::getMeshDimensions() - 1 ], coarseErrors );
         this->getApproximationError( 2 * coarseMeshSize[ MeshType::getMeshDimensions() - 1 ], fineErrors );
         this->checkEoc( coarseErrors, fineErrors, eoc, tolerance, verbose );
      }
 
   protected:

 

};


template< typename Mesh,
          typename Function,
          bool write,
          bool verbose >
bool runTest()
{
   typedef tnlOneSidedMeanCurvature< Mesh > ApproximateOperator;
   typedef tnlOneSidedMeanCurvatureTest< ApproximateOperator, Function, write, verbose > OperatorTest;
   OperatorTest test;
   test.runUnitTest();
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< tnlPDEOperatorEocUnitTest< OperatorTest > >() )
      return false;
   return true;
#else
   return false;
#endif
}

template< typename Mesh,
          bool write,
          bool verbose >
bool setTestFunction()
{
   return runTest< Mesh, tnlExpBumpFunction< Mesh::getMeshDimensions(), double >, write, verbose >();
}

template< typename Device,
          bool write,
          bool verbose >
bool setMesh()
{
   return ( setTestFunction< tnlGrid< 1, double, Device, int >, write, verbose >() &&
            setTestFunction< tnlGrid< 2, double, Device, int >, write, verbose >() &&
            setTestFunction< tnlGrid< 3, double, Device, int >, write, verbose >() );
}

int main( int argc, char* argv[] )
{
   const bool verbose( true );
   const bool write( true );
 
   if( ! setMesh< tnlHost, write, verbose  >() )
      return EXIT_FAILURE;
#ifdef HAVE_CUDA
   if( ! setMesh< tnlCuda, write, verbose >() )
      return EXIT_FAILURE;
#endif
   return EXIT_SUCCESS;
}

#endif	/* TNLMEANCURVATURETEST_H */

