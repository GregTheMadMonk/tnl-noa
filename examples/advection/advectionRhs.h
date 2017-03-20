#pragma once

#include <TNL/Functions/Domain.h>

namespace TNL {

template< typename Mesh, typename Real >class advectionRhs
  : public Functions::Domain< Mesh::meshDimensions, Functions::MeshDomain > 
{
   public:

      typedef Mesh MeshType;
      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true;
      }

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         typedef typename MeshEntity::MeshType::VertexType VertexType;
         VertexType v = entity.getCenter();
         return 0.0;
      }
};

} // namespace TNL

