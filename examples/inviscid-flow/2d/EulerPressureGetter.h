#ifndef EulerPressureGetter_H
#define EulerPressureGetter_H

<<<<<<< HEAD
#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlDomain.h>
=======
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
>>>>>>> develop

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerPressureGetter
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
class EulerPressureGetter< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
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
=======
      static String getType();
      Real gamma;
      MeshFunctionType velocity;
      MeshFunctionType rho;
      MeshFunctionType energy;

      void setGamma(const Real& gamma)
      {
          this->gamma = gamma;
      };

      void setVelocity(const MeshFunctionType& velocity)
      {
          this->velocity = velocity;
      };

      void setRho(const MeshFunctionType& rho)
      {
          this->rho = rho;
      };

      void setEnergy(const MeshFunctionType& energy)
      {
          this->energy = energy;
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
class EulerPressureGetter< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
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
      Real gamma;
      MeshFunctionType velocity;
      MeshFunctionType rho;
      MeshFunctionType energy;

      void setGamma(const Real& gamma)
      {
          this->gamma = gamma;
      };

      void setVelocity(const MeshFunctionType& velocity)
      {
          this->velocity = velocity;
      };

      void setRho(const MeshFunctionType& rho)
      {
          this->rho = rho;
      };

      void setEnergy(const MeshFunctionType& energy)
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
class EulerPressureGetter< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
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
      Real gamma;
      MeshFunctionType velocity;
      MeshFunctionType rho;
      MeshFunctionType energy;

      void setGamma(const Real& gamma)
>>>>>>> develop
      {
/*         if (this->rho[ idx ]==0) return 0; 
         else return ( this->gamma - 1.0 ) * ( this->energy[ idx ] - 0.5 * this->rho[ idx ] * 
         ( this->rhoVelX[ idx ] / this->rho[ idx ] + this->rhoVelY[ idx ] / this->rho[ idx ]) );
*/       return ( this->gamma - 1.0 ) * ( this->energy[ idx ] * this->rho[ idx ] );
      }

      
   protected:
      
      Real gamma;
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVelX;
      
      const MeshFunctionType& rhoVelY;

<<<<<<< HEAD
      const MeshFunctionType& energy;
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

#include "EulerPressureGetter_impl.h"

>>>>>>> develop
#endif	/* EulerPressureGetter_H */
