/***************************************************************************
                          tnlCoFVMGradientNormTest.h  -  description
                             -------------------
    begin                : Jan 17, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLTWOSIDEDGRADIENTNORMTEST_H
#define	TNLTWOSIDEDGRADIENTNORMTEST_H

#include <operators/geometric/tnlCoFVMGradientNorm.h>
#include <operators/geometric/tnlExactGradientNorm.h>
#include <operators/interpolants/tnlMeshEntitiesInterpolants.h>
#include <operators/tnlOperatorComposition.h>
#include "../../tnlUnitTestStarter.h"
#include "../tnlPDEOperatorEocTest.h"
#include "../tnlPDEOperatorEocUnitTest.h"

using namespace TNL;

template< typename ApproximateOperator,
          typename TestFunction,
          bool write = false,
          bool verbose = false >
class tnlCoFVMGradientNormTest
   : public tnlPDEOperatorEocTest< ApproximateOperator, TestFunction >
{
   public:
 
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename ApproximateOperatorType::ExactOperatorType ExactOperatorType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::IndexType IndexType;
 
      const IndexType coarseMeshSize[ 3 ] = { 1024, 256, 64 };

      const RealType eoc[ 3 ] =       { 2.0,  2.0, 2.0 };
      const RealType tolerance[ 3 ] = { 1.05, 1.1, 1.3 };
 
      static tnlString getType()
      {
         return tnlString( "tnlCoFVMGradientNormTest< " ) +
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
 
         tnlMeshFunction< MeshType > u;
 
         typename ApproximateOperator::InnerOperator gradientNorm;
         typename ApproximateOperator::OuterOperator interpolant;
         ApproximateOperator approximateOperator( interpolant, gradientNorm, this->mesh );
         typename ApproximateOperator::InnerOperator::ExactOperatorType exactOperator;

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
         this->checkEoc( coarseErrors, fineErrors, this->eoc, this->tolerance, verbose );
      }
 
   protected:


};


template< typename Operator,
          typename Function,
          bool write,
          bool verbose >
bool runTest()
{
   typedef tnlCoFVMGradientNormTest< Operator, Function, write, verbose > OperatorTest;
#ifdef HAVE_CPPUNIT
   if( ! tnlUnitTestStarter::run< tnlPDEOperatorEocUnitTest< OperatorTest > >() )
      return false;
   return true;
#endif
}

template< typename Mesh,
          typename Function,
          bool write,
          bool verbose >
bool setDifferenceOperator()
{
   typedef tnlCoFVMGradientNorm< Mesh > GradientNorm;
   return ( runTest< GradientNorm, Function, write, verbose >() );
}

template< typename Mesh,
          bool write,
          bool verbose >
bool setTestFunction()
{
   return setDifferenceOperator< Mesh, tnlExpBumpFunction< Mesh::getMeshDimensions(), double >, write, verbose >();
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
   const bool write( false );
 
   if( ! setMesh< tnlHost, write, verbose  >() )
      return EXIT_FAILURE;
#ifdef HAVE_CUDA
   if( ! setMesh< tnlCuda, write, verbose >() )
      return EXIT_FAILURE;
#endif
   return EXIT_SUCCESS;
}


#endif	/* TNLFDMGRADIENTNORMTEST_H */

