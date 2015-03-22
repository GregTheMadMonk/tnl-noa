#ifndef TNLOPERATORQ_H
#define	TNLOPERATORQ_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>
#include "tnlBackwardFiniteDifference.h"
#include "tnlForwardFiniteDifference.h"

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlOperatorQ
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOperatorQ< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
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
   Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const 
   {}
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const 
   {}
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
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
   Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const;
   
   private:
   
   tnlForwardFiniteDifference<MeshType> fDifference;
   tnlBackwardFiniteDifference<MeshType> bDifference;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
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
   Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const 
   {}
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
                   const IndexType cellIndex,
                   const Vector& u) const 
   {}
};


#include "tnlOperatorQ_impl.h"


#endif	/* TNLOPERATORQ_H */
