/***************************************************************************
                          tnlLinearDiffusion.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Vectors/Vector.h>
#include <TNL/Functions/tnlMeshFunction.h>
#include <TNL/mesh/tnlGrid.h>
#include <TNL/operators/tnlOperator.h>
#include <TNL/operators/diffusion/tnlExactLinearDiffusion.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlLinearDiffusion
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
: public tnlOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >,
                      MeshInteriorDomain, 1, 1, Real, Index >
{
   public:
 
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlExactLinearDiffusion< 1 > ExactOperatorType;
 
      static const int Dimensions = MeshType::meshDimensions;
 
      static constexpr int getMeshDimensions() { return Dimensions; }
 
      static String getType();

      template< typename PreimageFunction,
                typename MeshEntity >
      __cuda_callable__
      inline Real operator()( const PreimageFunction& u,
                              const MeshEntity& entity,
                              const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      inline Index getLinearSystemRowLength( const MeshType& mesh,
                                             const IndexType& index,
                                             const MeshEntity& entity ) const;

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      inline void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
: public tnlOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >,
                      MeshInteriorDomain, 2, 2, Real, Index >
{
   public:
 
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlExactLinearDiffusion< 2 > ExactOperatorType;
 
      static const int Dimensions = MeshType::meshDimensions;
 
      static constexpr int getMeshDimensions() { return Dimensions; }

      static String getType();

      template< typename PreimageFunction, typename EntityType >
      __cuda_callable__
      inline Real operator()( const PreimageFunction& u,
                              const EntityType& entity,
                              const Real& time = 0.0 ) const;

      template< typename EntityType >
      __cuda_callable__
      inline Index getLinearSystemRowLength( const MeshType& mesh,
                                             const IndexType& index,
                                             const EntityType& entity ) const;
 
      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      inline void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
: public tnlOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >,
                      MeshInteriorDomain, 3, 3, Real, Index >
{
   public:
 
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlExactLinearDiffusion< 3 > ExactOperatorType;

      static const int Dimensions = MeshType::meshDimensions;
 
      static constexpr int getMeshDimensions() { return Dimensions; }

      static String getType();

      template< typename PreimageFunction,
                typename EntityType >
      __cuda_callable__
      inline Real operator()( const PreimageFunction& u,
                              const EntityType& entity,
                              const Real& time = 0.0 ) const;

      template< typename EntityType >
      __cuda_callable__
      inline Index getLinearSystemRowLength( const MeshType& mesh,
                                             const IndexType& index,
                                             const EntityType& entity ) const;

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      inline void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const;
};

} //namespace TNL

#include <TNL/operators/diffusion/tnlLinearDiffusion_impl.h>
