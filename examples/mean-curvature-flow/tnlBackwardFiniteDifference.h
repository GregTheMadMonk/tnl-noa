#ifndef TNLBACKWARDFINITEDIFFERENCE_H
#define	TNLBACKWARDFINITEDIFFERENCE_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlBackwardFiniteDifference
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlBackwardFiniteDifference< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueX( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const 
   {}
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueY( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const
   {}
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlBackwardFiniteDifference< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType(); 

   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueX( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueY( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlBackwardFiniteDifference< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueX( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const 
   {}
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueY( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const
   {}
};


#include "tnlBackwardFiniteDifference_impl.h"


#endif	/* TNLBACKWARDFINITEDIFFERENCE_H */
