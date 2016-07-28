/***************************************************************************
                          tnlIdentityOperator.h  -  description
                             -------------------
    begin                : Feb 9, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/tnlMeshFunction.h>

namespace TNL {

template< typename MeshFunction >
class tnlIdentityOperator
   : public tnlDomain< MeshFunction::getDimensions(), MeshFunction::getDomainType() >
{
   public:
 
      typedef typename MeshFunction::MeshType MeshType;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::IndexType IndexType;
 
      tnlOperatorComposition( const MeshFunction& meshFunction )
      : meshFunction( meshFunction ) {};
 
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshFunction& function,
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         static_assert( MeshFunction::getDimensions() == InnerOperator::getDimensions(),
            "Mesh function and operator have both different number of dimensions." );
         return this->meshFunction( meshEntity, time );
      }
 
 
   protected:
 
      const MeshFunction& meshFunction;
};

} // namespace TNL

