/***************************************************************************
                          tnlNeumannBoundaryConditions.h  -  description
                             -------------------
    begin                : Feb 17, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLNEUMANNBOUNDARYCONDITIONS_H
#define	TNLNEUMANNBOUNDARYCONDITIONS_H

template< typename Mesh,
          typename Vector,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlNeumannBoundaryConditions
{

};

/****
 * Base
 */
template< typename Vector >
class tnlNeumannBoundaryConditionsBase
{
   public:

      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" );

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );

      Vector& getVector();

      const Vector& getVector() const;

   protected:

      Vector vector;

};

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Vector >
{
   public:

   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Vector VectorType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

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

   template< typename MatrixRow >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;
};

/****
 * 2D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Vector >
{
   public:

   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Vector VectorType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

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

   template< typename MatrixRow >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;
};

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Vector >
{
   public:

   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Vector VectorType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

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

   template< typename MatrixRow >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;
};

template< typename Mesh,
          typename Function,
          typename Real,
          typename Index >
ostream& operator << ( ostream& str, const tnlNeumannBoundaryConditions< Mesh, Function, Real, Index >& bc )
{
   str << "Dirichlet boundary conditions: vector = " << bc.getVector();
   return str;
}


#include <operators/tnlNeumannBoundaryConditions_impl.h>

#endif	/* TNLNEUMANNBOUNDARYCONDITIONS_H */

