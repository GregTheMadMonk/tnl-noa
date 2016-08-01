#ifndef EulerVelYGetter_H
#define EulerVelYGetter_H

#include <TNL/Vectors/Vector.h>
#include <TNL/mesh/tnlGrid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelYGetter
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class EulerVelYGetter< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static String getType();
      MeshFunctionType rhoVelY;
      MeshFunctionType rho;

      void setRhoVelY(const MeshFunctionType& rhoVelY)
      {
          this->rhoVelY = rhoVelY;
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
class EulerVelYGetter< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static String getType();
      MeshFunctionType rhoVelY;
      MeshFunctionType rho;

      void setRhoVelY(const MeshFunctionType& rhoVelY)
      {
          this->rhoVelY = rhoVelY;
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
class EulerVelYGetter< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      static String getType();
      MeshFunctionType rhoVelY;
      MeshFunctionType rho;

      void setRhoVelY(const MeshFunctionType& rhoVelY)
      {
          this->rhoVelY = rhoVelY;
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

} //namespace TNL


#include "EulerVelYGetter_impl.h"

#endif	/* EulerVelYGetter_H */
