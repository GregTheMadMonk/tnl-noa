#ifndef advectionRHS_H_
#define advectionRHS_H_
#include <TNL/functions/tnlDomain.h>
template< typename Mesh, typename Real >class advectionRhs
  : public tnlDomain< Mesh::meshDimensions, MeshDomain > 
 {
   public:

      typedef Mesh MeshType;
      typedef Real RealType;

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" )
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

#endif /* advectionRHS_H_ */
