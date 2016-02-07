/***************************************************************************
                          tnlPDEOperatorEocTest.h  -  description
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

#ifndef TNLPDEOPERATOREOSTEST_H
#define	TNLPDEOPERATOREOSTEST_H

#include "tnlPDEOperatorEocTestMeshSetter.h"
#include "tnlPDEOperatorEocTestFunctionSetter.h"
#include "tnlApproximationError.h"

template< typename ApproximateOperator,
          typename TestFunction >
class tnlPDEOperatorEocTest
{
   public:      
      
      typedef ApproximateOperator ApproximateOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename ApproximateOperator::ExactOperatorType ExactOperatorType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::IndexType IndexType;
   
      void setupMesh( const IndexType meshSize )
      {
         tnlPDEOperatorEocTestMeshSetter< MeshType >::setup( mesh, meshSize );
      }
      
      void setupFunction()
      {
         tnlPDEOperatorEocTestFunctionSetter< TestFunction >::setup( function );
      }
     
      template< typename MeshEntityType = typename MeshType::Cell > 
      void performTest( ApproximateOperator& approximateOperator,
                        ExactOperatorType& exactOperator,
                        RealType errors[ 3 ],
                        bool writeFunctions = false,
                        bool verbose = false )
      {
         tnlApproximationError< ExactOperatorType,
                                ApproximateOperator,
                                MeshEntityType,
                                TestFunction >
         ::getError( exactOperator,
                     approximateOperator,
                     function,
                     mesh,
                     errors[ 1 ], //l1Error,
                     errors[ 2 ], //l2Error,
                     errors[ 0 ], //maxError,
                     writeFunctions );
         if( verbose )
            std::cout << "L1 err. " << errors[ 1 ] << " L2 err. " << errors[ 2 ] << " Max. err. " << errors[ 0 ] << std::endl;
      }
      
      void checkEoc( const RealType coarse[ 3 ],
                     const RealType fine[ 3 ],
                     const RealType eoc[ 3 ],
                     const RealType tolerance[ 3 ],
                     bool verbose = false)
      {
         /****
          * Max error
          */
         RealType maxEoc = log( coarse[ 0 ] / fine[ 0 ] ) / log( 2.0 );
         if( verbose )
            std::cout << "Max error EOC = " << maxEoc << std::endl;
#ifdef HAVE_CPPUNIT
         CPPUNIT_ASSERT( fabs( maxEoc - eoc[ 0 ] ) < tolerance[ 0 ] );
#endif
         
         /****
          * Max error
          */
         RealType l1Eoc = log( coarse[ 1 ] / fine[ 1 ] ) / log( 2.0 );
         if( verbose )
            std::cout << "L1 error EOC = " << l1Eoc << std::endl;         
#ifdef HAVE_CPPUNIT
         CPPUNIT_ASSERT( fabs( l1Eoc - eoc[ 1 ] ) < tolerance[ 1 ] );
#endif         
         
         /****
          * L2 error
          */
         RealType l2Eoc = log( coarse[ 2 ] / fine[ 2 ] ) / log( 2.0 );
         if( verbose )
            std::cout << "L2 error EOC = " << l2Eoc << std::endl;
#ifdef HAVE_CPPUNIT         
         CPPUNIT_ASSERT( fabs( l2Eoc - eoc[ 2 ] ) < tolerance[ 2 ] );
#endif         
      }
      
   protected:
      
      MeshType mesh;
      
      TestFunction function;
         
      
};


#endif	/* TNLPDEOPERATOREOSTEST_H */

