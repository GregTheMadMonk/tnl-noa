#ifndef eulerRHS_H_
#define eulerRHS_H_

#include <TNL/functions/tnlDomain.h>

namespace TNL {

template< typename Mesh, typename Real >class eulerRhs
  : public tnlDomain< Mesh::meshDimensions, MeshDomain > 
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
      };
};

} //namespace TNL

#endif /* eulerRHS_H_ */
