/***************************************************************************
                          tnlOneSidedMeanCurvatureTest.h  -  description
                             -------------------
    begin                : Feb 1, 2016
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

#ifndef TNLMEANCURVATURETEST_H
#define	TNLMEANCURVATURETEST_H

#include <operators/diffusion/tnlOneSidedMeanCurvature.h>
#include <operators/diffusion/tnlExactLinearDiffusion.h>
#include <mesh/tnlGrid.h>
#include "../tnlPDEOperatorEocUnitTest.h"
#include "../tnlPDEOperatorEocTest.h"
#include "../../tnlUnitTestStarter.h"

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
      
      const IndexType coarseMeshSize[ 3 ] = { 1024, 256, 64 };
      
      const RealType  eoc[ 3 ] =       { 2.0,  2.0,  2.0 };
      const RealType  tolerance[ 3 ] = { 0.05, 0.05, 0.05 };      
   
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
         this->performTest( this->approximateOperator,
                            this->exactOperator,
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

      ApproximateOperator approximateOperator;
      
      ExactOperatorType exactOperator;

};


template< typename Mesh,
          typename Function,
          bool write,
          bool verbose >
bool runTest()
{
   typedef tnlOneSidedMeanCurvature< Mesh > ApproximateOperator;
   typedef tnlOneSidedMeanCurvatureTest< ApproximateOperator, Function, write, verbose > OperatorTest;
#ifdef HAVE_CPPUNIT   
   if( ! tnlUnitTestStarter::run< tnlPDEOperatorEocUnitTest< OperatorTest > >() )
      return false;
   return true;
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
   const bool verbose( false );
   const bool write( false );
   
   if( ! setMesh< tnlHost, write, verbose  >() )
      return EXIT_FAILURE;
#ifdef HAVE_CUDA
   if( ! setMesh< tnlCuda, write, verbose >() )
      return EXIT_FAILURE;
#endif   
   return EXIT_SUCCESS;
}

#endif	/* TNLMEANCURVATURETEST_H */

