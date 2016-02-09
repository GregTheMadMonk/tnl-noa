/***************************************************************************
                          tnlCoFVMGradientNormTest.h  -  description
                             -------------------
    begin                : Jan 17, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLTWOSIDEDGRADIENTNORMTEST_H
#define	TNLTWOSIDEDGRADIENTNORMTEST_H

#include <operators/geometric/tnlCoFVMGradientNorm.h>
#include <operators/geometric/tnlExactGradientNorm.h>
#include <operators/interpolants/tnlMeshEntitiesInterpolants.h>
#include <operators/tnlOperatorComposition.h>
#include "../../tnlUnitTestStarter.h"
#include "../tnlPDEOperatorEocTest.h"
#include "../tnlPDEOperatorEocUnitTest.h"

template< typename ApproximateOperator,
          typename TestFunction,
          bool write = false,
          bool verbose = false >
class tnlCoFVMGradientNormTest
   : public tnlPDEOperatorEocTest< 
      tnlOperatorComposition<
         tnlMeshEntitiesInterpolants< 
            typename ApproximateOperator::MeshType,
            ApproximateOperator::MeshType::getDimensionsCount() - 1,
            ApproximateOperator::MeshType::getDimensionsCount() >,
         ApproximateOperator >,
      TestFunction,
      typename ApproximateOperator::ExactOperatorType >
{
   public:
      
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename ApproximateOperatorType::ExactOperatorType ExactOperatorType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::IndexType IndexType;
      
      const IndexType coarseMeshSize[ 3 ] = { 1024, 256, 64 };

      const RealType eoc[ 3 ] =       { 1.0,  1.9, 1.75 };
      const RealType tolerance[ 3 ] = { 0.05, 0.1, 0.3 };      
      
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
         
         typedef tnlMeshFunction< MeshType, MeshType::getDimensionsCount()> DiscreteTestFunction;
         DiscreteTestFunction discreteTestFunction( this->mesh );
         discreteTestFunction = this->function;
         
         typedef tnlOperatorFunction< ApproximateOperator, DiscreteTestFunction > GradientNormOnFacesOperator;
         GradientNormOnFacesOperator gradientNormOnFacesOperator( this->approximateOperator, discreteTestFunction );
         
         typedef tnlMeshFunction< MeshType, MeshType::getDimensionsCount() - 1 > GradientNormOnFacesFunction;
         GradientNormOnFacesFunction gradientNormOnFacesFunction( this->mesh );
         gradientNormOnFacesFunction = gradientNormOnFacesOperator;
                  
         typedef tnlMeshEntitiesInterpolants< MeshType, MeshType::getDimensionsCount() - 1 , MeshType::getDimensionsCount() > Interpolant;
         Interpolant interpolant;
         
         // TODOL udelat implementaci iterpolace pro operatory a specializaci pro mesh functions
         
         this->performTest( testOperator,
                            this->exactOperator,
                            errors,
                            write,
                            verbose );
      }
      
      void runUnitTest()
      {  
         RealType coarseErrors[ 3 ], fineErrors[ 3 ];
         this->getApproximationError( coarseMeshSize[ MeshType::getDimensionsCount() - 1 ], coarseErrors );
         this->getApproximationError( 2 * coarseMeshSize[ MeshType::getDimensionsCount() - 1 ], fineErrors );
         this->checkEoc( coarseErrors, fineErrors, this->eoc, this->tolerance, verbose );                            
      }
      
   protected:

      ApproximateOperator approximateOperator;
      
      ExactOperatorType exactOperator;

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
   return setDifferenceOperator< Mesh, tnlExpBumpFunction< Mesh::getDimensionsCount(), double >, write, verbose >();
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

