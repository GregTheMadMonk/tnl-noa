#ifndef EulerPressureGetter_H
#define EulerPressureGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Domain.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerPressureGetter
: public Functions::Domain< Mesh::getMeshDimensions(), Functions::MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static String getType();
      
      EulerPressureGetter( const MeshFunctionType& velocity,
                           const MeshFunctionType& rhoVel,
                           const MeshFunctionType& energy,
                           const RealType& gamma )
      : rho( rho ), rhoVel( rhoVel ), energy( energy ), gamma( gamma )
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
         return ( this->gamma - 1.0 ) * ( this->energy[ idx ] - 0.5 * this->rhoVel[ idx ] * this->rhoVel[ idx ] / this->rho[ idx ]);
      }

      
   protected:
      
      Real gamma;
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVel;
      
      const MeshFunctionType& energy;

};

} //namespace TNL

#endif	/* EulerPressureGetter_H */
