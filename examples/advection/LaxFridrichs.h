#ifndef LaxFridrichs_H
#define LaxFridrichs_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichs
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichs< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimension = MeshType::getMeshDimension() };
      Real tau;
      Real artificalViscosity;
      Real advectionSpeedX;
      Real advectionSpeedY;

      void setAdvectionSpeedY(const Real& advectionSpeed)
      {
	   this->advectionSpeedY = advectionSpeed;
      }


      void setAdvectionSpeedX(const Real& advectionSpeed)
      {
	   this->advectionSpeedX = advectionSpeed;
      }

      void setViscosity(const Real& artificalViscosity)
      {
	   this->artificalViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      static String getType();

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
class LaxFridrichs< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimension = MeshType::getMeshDimension() };
      Real tau;
      Real artificalViscosity;
      Real advectionSpeedX;
      Real advectionSpeedY;

      void setAdvectionSpeedY(const Real& advectionSpeed)
      {
	   this->advectionSpeedY = advectionSpeed;
      }


      void setAdvectionSpeedX(const Real& advectionSpeed)
      {
	   this->advectionSpeedX = advectionSpeed;
      }

      void setViscosity(const Real& artificalViscosity)
      {
	   this->artificalViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      static String getType();

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
class LaxFridrichs< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimension = MeshType::getMeshDimension() };
      Real tau;
      Real artificalViscosity;
      Real advectionSpeedX;
      Real advectionSpeedY;

      void setAdvectionSpeedY(const Real& advectionSpeed)
      {
	   this->advectionSpeedY = advectionSpeed;
      }


      void setAdvectionSpeedX(const Real& advectionSpeed)
      {
	   this->advectionSpeedX = advectionSpeed;
      }

      void setViscosity(const Real& artificalViscosity)
      {
	   this->artificalViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      static String getType();

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

} // namespace TNL


#include "LaxFridrichs_impl.h"

#endif	/* LaxFridrichs_H */
