#ifndef BenchmarkLaplace_H
#define BenchmarkLaplace_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class BenchmarkLaplace
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class BenchmarkLaplace< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
: public Operators::Operator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                              Functions::MeshInteriorDomain,
                              Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions(),
                              Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions(),
                              Real,
                              Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

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
      void setMatrixElements( const RealType& time,
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
class BenchmarkLaplace< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
: public Operators::Operator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >,
                              Functions::MeshInteriorDomain,
                              Meshes::Grid< 2, MeshReal, Device, MeshIndex >::getMeshDimensions(),
                              Meshes::Grid< 2, MeshReal, Device, MeshIndex >::getMeshDimensions(),
                              Real,
                              Index >
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
      void setMatrixElements( const RealType& time,
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
class BenchmarkLaplace< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
: public Operators::Operator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >,
                              Functions::MeshInteriorDomain,
                              Meshes::Grid< 3, MeshReal, Device, MeshIndex >::getMeshDimensions(),
                              Meshes::Grid< 3, MeshReal, Device, MeshIndex >::getMeshDimensions(),
                              Real,
                              Index >
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
      void setMatrixElements( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};


#include "BenchmarkLaplace_impl.h"

#endif	/* BenchmarkLaplace_H */
