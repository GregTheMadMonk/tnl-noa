/***************************************************************************
                          tnlLinearDiffusionTest.h  -  description
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

#ifndef TNLLINEARDIFFUSIONTEST_H
#define	TNLLINEARDIFFUSIONTEST_H

#include <operators/diffusion/tnlLinearDiffusion.h>
#include <operators/diffusion/tnlExactLinearDiffusion.h>
#include <mesh/tnlGrid.h>
#include "../tnlPDEOperatorEocUnitTest.h"

/*template< int Dimensions,
          typedef Real,
          typedef Device,
          typedef Index,
          typedef Function >
class tnlPDEOperatorEocTest< tnlLinearDiffusion< tnlGrid< Dimensions, Real, Device, Index >, Function >
{
   public:
      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef tnlLinearDiffusion< MeshType > ApproximateOperator;
      typedef tnlExactLinearDiffusion< Dimensions > ExactOperator;
      
      void setupTest() {};
      
      void runUnitTest()
      {
         
      }
      
      protected:
         
         MeshType mesh;
         FunctionType function;
};
 * */

template< typename ApproximateOperator,
          typename TestFunction >
class tnlLinearDiffusionTest
   : public tnlPDEOperatorEocTest< ApproximateOperator, TestFunction > 
{
   public:
      
};


template< typename Mesh,
          typename Function >
bool runTest()
{
   typedef tnlLinearDiffusion< Mesh > ApproximateOperator;
   typedef tnlLinearDiffusionTest< ApproximateOperator, Function > OperatorTest;
#ifdef HAVE_CPPUNIT   
   if( ! tnlUnitTestStarter::run< OperatorTest >() )
      return false;
   return true;
#endif      
}

template< typename Mesh >
bool setTestFunction()
{
   return runTest< Mesh, tnlExpBumpFunction< Mesh::getDimensionsCount(), double >();
}

template< typename Device >
bool setMesh()
{
   return ( setTestFunction< tnlGrid< 1, double, Device, int >() &&
      setTestFunction< tnlGrid< 2, double, Device, int >() &&
      setTestFunction< tnlGrid< 3, double, Device, int >() );
}

int main( int argc, char* argv[] )
{
   if( ! setMesh< tnlHost >() )
      return EXIT_FAILURE;
#ifdef HAVE_CUDA
   if( ! setMesh< tnlCuda >() )
      return EXIT_FAILURE;
#endif   
   return EXIT_SUCCESS;
}

#endif	/* TNLLINEARDIFFUSIONTEST_H */

