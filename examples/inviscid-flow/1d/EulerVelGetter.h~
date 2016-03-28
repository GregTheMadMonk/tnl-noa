#ifndef EulerVelGetter_H
#define EulerVelGetter_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelGetter
: public tnlDomain< Mesh::getMeshDimensions(), MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static tnlString getType();
      
      EulerVelGetter( const MeshFunctionType& rho,
                      const MeshFunctionType& rhoVel)
      : rho( rho ), rhoVel( rhoVel )
      {}

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         return this->operator[]( entity.getIndex() );
      }
      
      template< typename MeshEntity >
      __cuda_callable__
      Real operator[]( const IndexType& idx ) const
      {
         return this->rho[ idx ] / this->rhoVel[ idx ];
      }

      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVel;

};

#endif	/* EulerVelGetter_H */
