#ifndef TNLFORWARDFINITEDIFFERENCE_H
#define	TNLFORWARDFINITEDIFFERENCE_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlForwardFiniteDifference
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlForwardFiniteDifference< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
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
class tnlForwardFiniteDifference< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
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
class tnlForwardFiniteDifference< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
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


#include "tnlForwardFiniteDifference_impl.h"


#endif	/* TNLFORWARDFINITEDIFFERENCE_H */
