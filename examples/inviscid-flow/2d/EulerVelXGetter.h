#ifndef EulerVelXGetter_H
#define EulerVelXGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelXGetter
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
class EulerVelXGetter< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
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
      
      EulerVelXGetter( const MeshFunctionType& rho,
                      const MeshFunctionType& rhoVel)
      : rho( rho ), rhoVel( rhoVel )
      {}

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
=======
      static String getType();
      MeshFunctionType rhoVelX;
      MeshFunctionType rho;

      void setRhoVelX(const MeshFunctionType& rhoVelX)
      {
          this->rhoVelX = rhoVelX;
      };

      void setRho(const MeshFunctionType& rho)
      {
          this->rho = rho;
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
class EulerVelXGetter< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
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
      MeshFunctionType rhoVelX;
      MeshFunctionType rho;

      void setRhoVelX(const MeshFunctionType& rhoVelX)
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
class EulerVelXGetter< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
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
      MeshFunctionType rhoVelX;
      MeshFunctionType rho;

      void setRhoVelX(const MeshFunctionType& rhoVelX)
>>>>>>> develop
      {
         if (this->rho[ idx ]==0) return 0; else return (this->rhoVel[ idx ] / this->rho[ idx ]);
      }

<<<<<<< HEAD
      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVel;
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
} // namespace TNL

#include "EulerVelXGetter_impl.h"

>>>>>>> develop
#endif	/* EulerVelXGetter_H */
