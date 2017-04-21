#ifndef EulerVelGetter_H
#define EulerVelGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelGetter
: public Functions::Domain< Mesh::getDimension(), Functions::MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimension = MeshType::getDimension() };

      static String getType();
      
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
      
      __cuda_callable__
      Real operator[]( const IndexType& idx ) const
      {
         return this->rho[ idx ] / this->rhoVel[ idx ];
      }

      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVel;

};

} // namespace TNL

#endif	/* EulerVelGetter_H */
