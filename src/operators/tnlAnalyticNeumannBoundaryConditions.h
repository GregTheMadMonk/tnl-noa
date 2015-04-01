/***************************************************************************
                          tnlAnalyticNeumannBoundaryConditions.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLANALYTICNEUMANNBOUNDARYCONDITIONS_H_
#define TNLANALYTICNEUMANNBOUNDARYCONDITIONS_H_


template< typename Mesh,
          typename Function = tnlConstantFunction< Mesh::Dimensions,
                                                   typename Mesh::RealType >,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlAnalyticNeumannBoundaryConditions
{

};

template< typename Function >
class tnlAnalyticNeumannBoundaryConditionsBase
{
   public:

      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" );

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );

      void setFunction( const Function& function );

      Function& getFunction();

      const Function& getFunction() const;

   protected:

      Function function;
};

/****
 * 1D grid
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlAnalyticNeumannBoundaryConditionsBase< Function >
{
   public:


   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlAnalyticNeumannBoundaryConditions< MeshType, Function, Real, Index > ThisType;

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
          typename Function,
          typename Real,
          typename Index >
class tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlAnalyticNeumannBoundaryConditionsBase< Function >
{
   public:

   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

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
          typename Function,
          typename Real,
          typename Index >
class tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlAnalyticNeumannBoundaryConditionsBase< Function >
{
   public:

   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

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
ostream& operator << ( ostream& str, const tnlAnalyticNeumannBoundaryConditions< Mesh, Function, Real, Index >& bc )
{
   str << "Analytic Dirichlet boundary conditions: function = " << bc.getFunction();
   return str;
}

#include <operators/tnlAnalyticNeumannBoundaryConditions_impl.h>

#endif /* TNLANALYTICNEUMANNBOUNDARYCONDITIONS_H_ */
