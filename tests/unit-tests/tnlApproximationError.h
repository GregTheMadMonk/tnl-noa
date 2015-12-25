/***************************************************************************
                          tnlApproximationError.h  -  description
                             -------------------
    begin                : Aug 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLAPPROXIMATIONERROR_H_
#define TNLAPPROXIMATIONERROR_H_

#include <mesh/tnlGrid.h>
#include <functions/tnlConstantFunction.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <solvers/pde/tnlExplicitUpdater.h>

class tnlExplicitApproximation
{
   public:

   static tnlString getType()
   {
      return tnlString( "tnlExplicitApproximation" );
   };
};

class tnlImplicitApproximation
{
   public:

   static tnlString getType()
   {
      return tnlString( "tnlImplicitApproximation" );
   };
};

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function,
          typename ApproximationMethod >
class tnlApproximationError
{
};

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
class tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function, tnlExplicitApproximation >
{
     public:

      typedef typename ApproximateOperator::RealType RealType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef tnlConstantFunction< MeshType::meshDimensions, RealType > ConstantFunctionType;
      typedef tnlDirichletBoundaryConditions< MeshType, Function  > BoundaryConditionsType;

      static void getError( const Mesh& mesh,
                            const ExactOperator& exactOperator,
                            const ApproximateOperator& approximateOperator,
                            const Function& function,
                            RealType& l1Err,
                            RealType& l2Err,
                            RealType& maxErr );
};

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
class tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function, tnlImplicitApproximation >
{
     public:

      typedef typename ApproximateOperator::RealType RealType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef tnlConstantFunction< MeshType::meshDimensions, RealType > ConstantFunctionType;
      typedef tnlDirichletBoundaryConditions< MeshType, Function  > BoundaryConditionsType;

      static void getError( const Mesh& mesh,
                            const ExactOperator& exactOperator,
                            const ApproximateOperator& approximateOperator,
                            const Function& function,
                            RealType& l1Err,
                            RealType& l2Err,
                            RealType& maxErr );
};

#include "tnlApproximationError_impl.h"

#endif /* TNLAPPROXIMATIONERROR_H_ */
