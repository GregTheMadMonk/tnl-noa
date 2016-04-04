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
      
      EulerVelGetter( const MeshFunctionType& rhoX,
                      const MeshFunctionType& rhoVelX,
                      const MeshFunctionType& rhoVelY)
      : rho( rho ), rhoVelX( rhoVelX ), rhoVelY( rhoVelY )
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
cout << idx << endl;
         if (this->rho[ idx ]==0) return 0; else return sqrt( pow( this->rhoVelX[ idx ] / this->rho[ idx ], 2) + pow( this->rhoVelY[ idx ] / this->rho[ idx ], 2) ) ;
      }

      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVelX;

      const MeshFunctionType& rhoVelY;

};

#endif	/* EulerVelGetter_H */
