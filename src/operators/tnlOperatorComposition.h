/***************************************************************************
                          tnlOperatorComposition.h  -  description
                             -------------------
    begin                : Jan 30, 2016
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

#ifndef TNLOPERATORCOMPOSITION_H
#define	TNLOPERATORCOMPOSITION_H

#include<functions/tnlOperatorFunction.h>
#include<functions/tnlMeshFunction.h>

template< typename OuterOperator,
          typename InnerOperator >
class tnlOperatorComposition
   : public tnlDomain< InnerOperator::getDimensions(), InnerOperator::getDomainType() >
{
      static_assert( is_same< typename OuterOperator::MeshType, typename InnerOperator::MeshType >::value,
         "Both operators have different mesh types." );
   public:
      
      typedef typename InnerOperator::MeshType MeshType;
      typedef tnlOperatorFunction< InnerOperator, tnlMeshFunction< MeshType, InnerOperator::getImageMeshEntitiesDimensions() > > InnerOperatorFunction;
      typedef typename InnerOperator::RealType RealType;
      typedef typename InnerOperator::IndexType IndexType;
      
      tnlOperatorComposition( const OuterOperator& outerOperator,
                              const InnerOperator& innerOperator )
      : outerOperator( &outerOperator ), innerOperator( &innerOperator ) {};
      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshFunction& function,
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         static_assert( MeshFunction::getDimensions() == InnerOperator::getDimensions(),
            "Mesh function and operator have both different number of dimensions." );
         InnerOperatorFunction innerOperatorFunction( innerOperator, function );
         return outerOperator( innerOperatorFunction, meshEntity, time );
      }
      
   
   protected:
      
      const OuterOperator& outerOperator;
      
      const InnerOperator& innerOperator;
      
};

#endif	/* TNLOPERATORCOMPOSITION_H */

