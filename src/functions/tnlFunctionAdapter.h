/***************************************************************************
                          tnlFunctionAdapter.h  -  description
                             -------------------
    begin                : Nov 28, 2014
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

#ifndef TNLFUNCTIONADAPTER_H_
#define TNLFUNCTIONADAPTER_H_

#include <functions/tnlConstantFunction.h>
#include <functions/tnlFunctionType.h>

template< typename Mesh,
          typename Function,
          int FunctionType = tnlFunctionType< Function >::Type >
class tnlFunctionAdapter
{
};

/****
 * General implementation:
 * - it passes both mesh entity center and mesh entity index
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, tnlGeneralFunction >
{
   public:

      typedef Mesh MeshType;
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;

      template< int MeshEntityDimension >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const RealType& time = 0.0 )
      {
         return function.getValue( mesh, //.template getEntityCenter< MeshEntityDimension >,
                                   index,
                                   time );
      }
};

/****
 * General implementation with specialization for grid functions
 *  - it takes grid coordinates for faster entity center evaluation
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >, Function, tnlGeneralFunction >
{
         public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef Function FunctionType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const CoordinatesType& coordinates,
                                const RealType& time = 0.0 )
      {
         return function.getValue( mesh, //.template getCellCenter< VertexType >( coordinates ),
                                   index,
                                   time );
      }
};

/****
 * Specialization for discrete functions:
 * - it passes only the mesh entity index
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, tnlDiscreteFunction >
{
   public:

      typedef Mesh MeshType;
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;

      template< int MeshEntityDimension >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const RealType& time = 0.0 )
      {
         return function.getValue( index,
                                   time );
      }
};

/****
 * Specialization for discrete functions:
 * - it passes only the mesh entity index
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >, Function, tnlDiscreteFunction >
{
   public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

      //template< int MeshEntityDimension >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const CoordinatesType& coordinates,
                                const RealType& time = 0.0 )
      {
         return function.getValue( index,
                                   time );
      }
};


/****
 * Specialization for analytic functions:
 * - it does not pass the mesh entity index
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, tnlAnalyticFunction >
{
   public:

      typedef Mesh MeshType;
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;

      template< int MeshEntityDimension >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const RealType& time = 0.0 )
      {
         return function.getValue( mesh.template getEntityCenter< MeshEntityDimension >,
                                   time );
      }
};

/****
 * Specialization for analytic grid functions:
 * - it does not pass the mesh entity index
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >, Function, tnlAnalyticFunction >
{
         public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef Function FunctionType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const CoordinatesType& coordinates,
                                const RealType& time = 0.0 )
      {
         return function.getValue( mesh.template getCellCenter< VertexType >( coordinates ),
                                   time );
      }
};

// TODO: Fix the specializations for the constant function.
#ifdef UNDEF
/****
 * Specialization for constant function
 *  - it does not ask the mesh for the mesh entity center
 */
template< typename Mesh,
          int FunctionDimensions,
          typename Real >
class tnlFunctionAdapter< Mesh, tnlConstantFunction< FunctionDimensions, Real >, tnlAnalyticFunction >
{
   public:

      typedef Mesh MeshType;
      typedef tnlConstantFunction< FunctionDimensions, Real > FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;

      template< int MeshEntityDimension >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const RealType& time = 0.0 )
      {
         VertexType v;
         return function.getValue( v, time );
      }
};

/****
 * Specialization for grids and constant function
 *  - it takes grid coordinates for faster entity center evaluation
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >,
                          tnlConstantFunction< Dimensions, Real >,
                          tnlAnalyticFunction >
{
   public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef tnlConstantFunction< Dimensions, Real > FunctionType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType index,
                                const CoordinatesType& coordinates,
                                const RealType& time = 0.0 )
      {
         VertexType v;
         return function.getValue( v, time );
      }
};

#endif /* UNDEF */

#endif /* TNLFUNCTIONADAPTER_H_ */
