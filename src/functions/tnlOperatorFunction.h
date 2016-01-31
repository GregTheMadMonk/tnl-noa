/***************************************************************************
                          tnlOperatorFunction.h  -  description
                             -------------------
    begin                : Dec 31, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLOPERATORFUNCTION_H
#define	TNLOPERATORFUNCTION_H

#include <type_traits>
#include <core/tnlCuda.h>

/***
 * This class evaluates given operator on given function.
 * The main role of this type is that the mesh function evaluator
 * evaluates this function only on the INTERIOR mesh entities.
 */
template< typename Operator,
          typename MeshFunction >
class tnlOperatorFunction : public tnlDomain< Operator::getDimensions(), Operator::getDomainType() >
{   
   public:
      
      static_assert( MeshFunction::getDomainType() == MeshDomain ||
                     MeshFunction::getDomainType() == MeshInteriorDomain ||
                     MeshFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh functions may be used in the operator function. Use tnlExactOperatorFunction instead of tnlOperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunction::MeshType >::value,
          "Both, operator and mesh function must be defined on the same mesh." );
      
      typedef Operator OperatorType;
      typedef MeshFunction FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      
      tnlOperatorFunction(
         const OperatorType& operator_,
         const FunctionType& function )
      :  operator_( operator_ ), function( function ){};
      
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return operator_( function, meshEntity, time );
      }
      
   protected:
      
      const Operator& operator_;
      
      const FunctionType& function;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

#endif	/* TNLOPERATORFUNCTION_H */

