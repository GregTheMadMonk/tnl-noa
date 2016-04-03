/***************************************************************************
                          tnlLinearDiffusion.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLLINEARDIFFUSION_H
#define	TNLLINEARDIFFUSION_H

#include <core/vectors/tnlVector.h>
#include <functions/tnlMeshFunction.h>
#include <mesh/tnlGrid.h>
#include <operators/tnlOperator.h>
#include <operators/diffusion/tnlExactLinearDiffusion.h>

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
      
      static tnlString getType();

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

      static tnlString getType();

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

      static tnlString getType();

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


#include <operators/diffusion/tnlLinearDiffusion_impl.h>


#endif	/* TNLLINEARDIFFUSION_H */
