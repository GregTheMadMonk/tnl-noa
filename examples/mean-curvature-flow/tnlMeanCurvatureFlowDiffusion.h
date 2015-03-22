#ifndef TNLMEANCURVATIVEFLOWDIFFUSION_H
#define	TNLMEANCURVATIVEFLOWDIFFUSION_H

#include <core/vectors/tnlVector.h>
#include "tnlForwardFiniteDifference.h"
#include "tnlBackwardFiniteDifference.h"
#include "tnlOperatorQ.h"
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlMeanCurvatureFlowDiffusion
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlMeanCurvatureFlowDiffusion< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
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
                  const RealType& time) const 
   {}
   
//   void setupDofs ( const MeshType& mesh )
//   {}
//   
//   template< typename Vector >
//   void computeFirstGradient( const MeshType& mesh,
//                              const RealType& time,
//                              const Vector u)
//   {}
   

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const
   {}

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
                               MatrixRow& matrixRow ) const
   {}

};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlMeanCurvatureFlowDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
//   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   static tnlString getType();
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getValue( const MeshType& mesh,
                  const IndexType cellIndex,
                  const CoordinatesType& coordinates,
                  const Vector& u,
                  const RealType& time) const;
   
//   void setupDofs ( const MeshType& mesh );
//   
//   template< typename Vector >
//   void computeFirstGradient( const MeshType& mesh,
//                              const RealType& time,
//                              const Vector u);
   
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const
   {}

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
                               MatrixRow& matrixRow ) const
   {}
   
   protected:
      
   tnlForwardFiniteDifference<MeshType> fDifference;
   tnlBackwardFiniteDifference<MeshType> bDifference;
   tnlOperatorQ<MeshType> operatorQ;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlMeanCurvatureFlowDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
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
                  const RealType& time) const
   {}
   
//   void setupDofs ( const MeshType& mesh )
//   {}
//   
//   template< typename Vector >
//   void computeFirstGradient( const MeshType& mesh,
//                              const RealType& time,
//                              const Vector u )
//   {}
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const CoordinatesType& coordinates ) const 
   {}

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
                               MatrixRow& matrixRow ) const 
   {}

};


#include "tnlMeanCurvatureFlowDiffusion_impl.h"


#endif	/* TNLMEANCURVATIVEFLOWDIFFUSION_H */
