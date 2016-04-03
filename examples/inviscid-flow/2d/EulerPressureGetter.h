#ifndef EulerPressureGetter_H
#define EulerPressureGetter_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlDomain.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerPressureGetter
: public tnlDomain< Mesh::getMeshDimensions(), MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static tnlString getType();
      
      EulerPressureGetter( const MeshFunctionType& rho,
                           const MeshFunctionType& rhoVelX,
                           const MeshFunctionType& rhoVelY,
                           const MeshFunctionType& energy,
                           const RealType& gamma )
      : rho( rho ), rhoVelX( rhoVelX ), rhoVelY( rhoVelY ), energy( energy ), gamma( gamma )
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
         return ( this->gamma - 1.0 ) * ( this->energy[ idx ] - 0.5 * this->rho[ idx ] * 
         ( this->rhoVelX[ idx ] / this->rho[ idx ] + this->rhoVelY[ idx ] / this->rho[ idx ]) );
      }

      
   protected:
      
      Real gamma;
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVelX;
      
      const MeshFunctionType& rhoVelY;

      const MeshFunctionType& energy;

};

#endif	/* EulerPressureGetter_H */
