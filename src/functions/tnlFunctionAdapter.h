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

template< typename Mesh,
          typename Function >
class tnlFunctionAdapter
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
      RealType getValue( const MeshType& mesh,
                         const FunctionType& function,
                         const IndexType index,
                         const RealType& time = 0.0 )
      {
         return function.getValue( mesh.template getEntityCenter< MeshEntityDimension >,
                                   time );
      }
};

/****
 * Specialization for constant function
 *  - it does not ask the mesh for the mesh entity center
 */

template< typename Mesh,
          int FunctionDimensions,
          typename Real >
class tnlFunctionAdapter< Mesh, tnlConstantFunction< FunctionDimensions, Real > >
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
      RealType getValue( const MeshType& mesh,
                         const FunctionType& function,
                         const IndexType index,
                         const RealType& time = 0.0 )
      {
         VertexType v;
         return function.getValue( v, time );
      }
};

/****
 * Specialization for grids
 *  - it takes grid coordinates for faster entity center evaluation
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >, Function >
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
      RealType getValue( const MeshType& mesh,
                         const FunctionType& function,
                         const IndexType index,
                         const CoordinatesType& coordinates,
                         const RealType& time = 0.0 )
      {
         return function.getValue( mesh.template getCellCenter( coordinates ),
                                   time );
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
                          tnlConstantFunction< Dimensions, Real > >
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
      RealType getValue( const MeshType& mesh,
                         const FunctionType& function,
                         const IndexType index,
                         const CoordinatesType& coordinates,
                         const RealType& time = 0.0 )
      {
         VertexType v;
         return function.getValue( v, time );
      }
};

/****
 * Specialization for mesh functions
 */
template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
class tnlFunctionAdapter< Mesh, tnlVector< Real, Device, Index > >
{
   public:
      typedef Mesh MeshType;
      typedef tnlVector< Real, Device, Index > FunctionType;
      typedef Real RealType;
      typedef Index IndexType;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      RealType getValue( const MeshType& mesh,
                         const FunctionType& function,
                         const IndexType index,
                         const RealType& time = 0.0 )
      {
         return function[ index ];
      }
};

/****
 * Specialization for mesh functions
 */
template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
class tnlFunctionAdapter< Mesh, tnlSharedVector< Real, Device, Index > >
   : public tnlFunctionAdapter< Mesh, tnlVector< Real, Device, Index > >
{
   public:
      typedef tnlSharedVector< Real, Device, Index > FunctionType;
};

/****
 * Specialization for mesh functions and grids
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >,
                          tnlVector< Real, Device, Index > >
{
   public:
      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef tnlVector< Real, Device, Index > FunctionType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      RealType getValue( const MeshType& mesh,
                         const FunctionType& function,
                         const IndexType index,
                         const CoordinatesType& coordinates,
                         const RealType& time = 0.0 )
      {
         return function[ index ];
      }
};

/****
 * Specialization for mesh functions and grids
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >,
                          tnlSharedVector< Real, Device, Index > >
   : public tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >,
                                tnlVector< Real, Device, Index > >
{
   public:
      typedef tnlSharedVector< Real, Device, Index > FunctionType;
};

#endif /* TNLFUNCTIONADAPTER_H_ */
