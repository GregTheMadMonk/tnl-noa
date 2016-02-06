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

template< typename ApproximateOperator,
          typename ExactOperator,
          typename TestFunction >
class tnlPDEOperatorEocTest
{
   public:      
      
      typedef ApproximateOperator ApproximateOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename ApproximateOperator::MeshType MeshType;
   
      template< typename Mesh >
      void setupMesh( Mesh& mesh )
      {
         tnlPDEOperatorEocTestMeshSetter( mesh );
      }
      
      template< typename Function >
      void setupFunction( Function& function )
      {
         tnlPDEOperatorEocTestFunctionSetter( function );
      }
};


#endif	/* TNLPDEOPERATOREOSTEST_H */

