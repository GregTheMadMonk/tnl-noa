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
#include <functions/tnlDomain.h>

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
: public tnlDomain< 1, MeshInteriorDomain >
{
   public:    
   
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;      
      
      static const int Dimensions = MeshType::meshDimensions;
      
      static constexpr int getMeshDimensions() { return Dimensions; }
      
      template< int EntityDimensions = Dimensions >
      using MeshFunction = tnlMeshFunction< MeshType, EntityDimensions >;

      static tnlString getType();

      template< typename MeshEntity >
      __cuda_callable__
      inline Real getValue( const MeshFunction< 1 >& u,
                            const MeshEntity& entity,
                            const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      inline Index getLinearSystemRowLength( const MeshType& mesh,
                                             const IndexType& index,
                                             const MeshEntity& entity ) const;

      template< typename MeshEntity,
                typename Vector,
                typename MatrixRow >
      __cuda_callable__
      inline void updateLinearSystem( const RealType& time,
                                      const RealType& tau,
                                      const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity,
                                      const MeshFunction< 1 >& u,
                                      Vector& b,
                                      MatrixRow& matrixRow ) const;

};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
: public tnlDomain< 2, MeshInteriorDomain >
{
   public: 
   
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;      
      
      static const int Dimensions = MeshType::meshDimensions;
      
      static constexpr int getMeshDimensions() { return Dimensions; }

      
      template< int EntityDimensions = Dimensions >
      using MeshFunction = tnlMeshFunction< MeshType, EntityDimensions >;      

      static tnlString getType();

      template< typename EntityType >
      __cuda_callable__
      inline Real getValue( const MeshFunction< 2 >& u,
                            const EntityType& entity,
                            const Real& time = 0.0 ) const;

      template< typename EntityType >
      __cuda_callable__
      inline Index getLinearSystemRowLength( const MeshType& mesh,
                                             const IndexType& index,
                                             const EntityType& entity ) const;

      template< typename Vector,
                typename MatrixRow,
                typename EntityType >
      __cuda_callable__
      inline void updateLinearSystem( const RealType& time,
                                      const RealType& tau,
                                      const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity,
                                      const MeshFunction< 2 >& u,
                                      Vector& b,
                                      MatrixRow& matrixRow ) const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
: public tnlDomain< 3, MeshInteriorDomain >
{
   public: 
   
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static const int Dimensions = MeshType::meshDimensions;
      
      static constexpr int getMeshDimensions() { return Dimensions; }      

      template< int EntityDimensions = Dimensions >
      using MeshFunction = tnlMeshFunction< MeshType, EntityDimensions >;

      
      static tnlString getType();

      template< typename EntityType >
      __cuda_callable__
      inline Real getValue( const MeshFunction< 3 >& u,
                            const EntityType& entity,
                            const Real& time = 0.0 ) const;

      template< typename EntityType >
      __cuda_callable__
      inline Index getLinearSystemRowLength( const MeshType& mesh,
                                             const IndexType& index,
                                             const EntityType& entity ) const;

      template< typename Vector,
                typename MatrixRow,
                typename EntityType >
      __cuda_callable__
      inline void updateLinearSystem( const RealType& time,
                                      const RealType& tau,
                                      const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity,
                                      const MeshFunction< 3 >& u,
                                      Vector& b,
                                      MatrixRow& matrixRow ) const;

};


#include <operators/diffusion/tnlLinearDiffusion_impl.h>


#endif	/* TNLLINEARDIFFUSION_H */
