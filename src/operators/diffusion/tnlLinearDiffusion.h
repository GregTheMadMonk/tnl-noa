#ifndef TNLLINEARDIFFUSION_H
#define	TNLLINEARDIFFUSION_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

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
                  const CoordinatesType& coordinates,
                  const Vector& u,
                  const RealType& time ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const;

   template< typename Vector, typename MatrixRow >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               Vector& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;

};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
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
                  const CoordinatesType& coordinates,
                  const Vector& u,
                  const Real& time ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const;

   template< typename Vector, typename MatrixRow >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               Vector& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
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
                  const CoordinatesType& coordinates,
                  const Vector& u,
                  const Real& time ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const;

   template< typename Vector, typename MatrixRow >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               Vector& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;

};


#include <implementation/operators/diffusion/tnlLinearDiffusion_impl.h>


#endif	/* TNLLINEARDIFFUSION_H */
