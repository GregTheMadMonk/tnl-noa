#ifndef EulerVelGetter_H
#define EulerVelGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Domain.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelGetter
<<<<<<< HEAD
: public tnlDomain< Mesh::getMeshDimensions(), MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
=======
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class EulerVelGetter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
    : public TNL::Functions::Domain< 1, TNL::Functions::MeshDomain >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
>>>>>>> develop
      typedef Real RealType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

<<<<<<< HEAD
      static tnlString getType();
      
      EulerVelGetter( const MeshFunctionType& rho,
                      const MeshFunctionType& rhoVelX,
                      const MeshFunctionType& rhoVelY)
      : rho( rho ), rhoVelX( rhoVelX ), rhoVelY( rhoVelY )
      {}

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
=======
      static String getType();
      MeshFunctionType velX;
      MeshFunctionType velY;

      void setVelX(const MeshFunctionType& velX)
      {
          this->velX = velX;
      };

      void setVelY(const MeshFunctionType& velY)
      {
          this->velY = velY;
      };

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class EulerVelGetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
    : public TNL::Functions::Domain< 2, TNL::Functions::MeshDomain >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static String getType();
      MeshFunctionType velX;
      MeshFunctionType velY;

      void setVelX(const MeshFunctionType& velX)
>>>>>>> develop
      {
         return this->operator[]( entity.getIndex() );
      }
      
      __cuda_callable__
<<<<<<< HEAD
      Real operator[]( const IndexType& idx ) const
=======
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class EulerVelGetter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
    : public TNL::Functions::Domain< 3, TNL::Functions::MeshDomain >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static String getType();
      MeshFunctionType velX;
      MeshFunctionType velY;

      void setVelX(const MeshFunctionType& velX)
      {
          this->velX = velX;
      };

      void setVelY(const MeshFunctionType& velY)
>>>>>>> develop
      {
         if (this->rho[ idx ]==0) return 0; else return sqrt( pow( this->rhoVelX[ idx ] / this->rho[ idx ], 2) + pow( this->rhoVelY[ idx ] / this->rho[ idx ], 2) ) ;
      }

      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVelX;

<<<<<<< HEAD
      const MeshFunctionType& rhoVelY;
=======
      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;
>>>>>>> develop

};

<<<<<<< HEAD
=======
} //namespace TNL

#include "EulerVelGetter_impl.h"

>>>>>>> develop
#endif	/* EulerVelGetter_H */
