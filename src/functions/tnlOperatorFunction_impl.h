/***************************************************************************
                          tnlOperatorFunction_impl.h  -  description
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

#ifndef TNLOPERATORFUNCTION_IMPL_H
#define	TNLOPERATORFUNCTION_IMPL_H


template< typename Operator,
          typename Function >
tnlOperatorFunction< Operator, Function >::
tnlOperatorFunction(
   const OperatorType& operator_,
   const FunctionType& function )
:  operator_( &operator_ ),
   function( &function )
{   
}

template< typename Operator,
          typename Function >
   template< typename MeshEntity >
__cuda_callable__
typename tnlOperatorFunction< Operator, Function >::RealType 
tnlOperatorFunction< Operator, Function >::operator()(
   const MeshEntity& meshEntity,
   const RealType& time )
{
   //static_assert( MeshEntity::entityDimensions == Operator::getMeshEntityDimensions(), "Wrong mesh entity dimensions." );
   tnlAssert( ! meshEntity.isBoundaryEntity(), 
      cerr << "Operator functions are defined only for interior mesh entities. Entity = " << meshEntity );
   return operator_->getValue( meshEntity, function->getData(), time );
}

#endif	/* TNLOPERATORFUNCTION_IMPL_H */

