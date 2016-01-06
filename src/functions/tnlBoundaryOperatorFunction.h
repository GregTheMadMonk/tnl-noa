/***************************************************************************
                          tnlBoundaryOperatorFunction.h  -  description
                             -------------------
    begin                : Jan 6, 2016
    copyright            : (C) 2016 by oberhuber
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

#ifndef TNLBOUNDARYOPERATORFUNCTION_H
#define	TNLBOUNDARYOPERATORFUNCTION_H

/***
 * This class evaluates given operator on given function.
 * The main role of this type is that the mesh function evaluator
 * evaluates this function only on the BOUNDARY mesh entities.
 */
template< typename BoundaryOperator,
          typename Function >
class tnlBoundaryOperatorFunction
{
   public:
      
      typedef BoundaryOperator BoundaryOperatorType;
      typedef Function FunctionType;
      typedef typename BoundaryOperator::MeshType MeshType;
      typedef typename BoundaryOperator::RealType RealType;
      
      tnlBoundaryOperatorFunction(
         const BoundaryOperatorType& boundaryOperator,
         const FunctionType& function )
      :  boundaryOperator( &boundaryOperator ), function( &function ){};
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time )
      {
         return boundaryOperator->getValue( meshEntity, function->getData(), time );
      }
      
   protected:
      
      const BoundaryOperator* boundaryOperator;
      
      const FunctionType* function;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};


#endif	/* TNLBOUNDARYOPERATORFUNCTION_H */

