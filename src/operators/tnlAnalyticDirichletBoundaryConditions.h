/***************************************************************************
           tnlAnalyticDirichletBoundaryConditions.h  -  description
                             -------------------
    begin                : Nov 8, 2014
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


#ifndef tnlAnalyticDirichletBoundaryConditions_H
#define	tnlAnalyticDirichletBoundaryConditions_H

#include <core/vectors/tnlStaticVector.h>
#include <config/tnlParameterContainer.h>
#include <functions/tnlConstantFunction.h>
#include <core/vectors/tnlSharedVector.h>

template< typename Mesh,
          typename Function = tnlConstantFunction< Mesh::Dimensions,
                                                   typename Mesh::RealType >,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlAnalyticDirichletBoundaryConditions
{
   
};

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >
{
   public:
   
   typedef tnlGrid< Dimensions, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< Dimensions, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );
            
   bool setup( const tnlParameterContainer& parameters,
               const tnlString& prefix = "" );

   void setFunction( const Function& function );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void setBoundaryConditions( const RealType& time,
                               const MeshType& mesh,
                               const IndexType index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& fu ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& b,
                               IndexType* columns,
                               RealType* values,
                               IndexType& rowLength ) const;

   protected:

   Function function;
};

#include <implementation/operators/tnlAnalyticDirichletBoundaryConditions_impl.h>

#endif	/* tnlAnalyticDirichletBoundaryConditions_H */
