#ifndef TNLNEUMANNREFLECTIONBOUNDARYCONDITIONS_H
#define	TNLNEUMANNREFLECTIONBOUNDARYCONDITIONS_H

#include <core/vectors/tnlStaticVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <config/tnlParameterContainer.h>
#include <functions/tnlConstantFunction.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlNeumannReflectionBoundaryConditions
{

};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlNeumannReflectionBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:

   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );


#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void setBoundaryConditions( const RealType& time,
                               const MeshType& mesh,
                               const IndexType index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& fu );


   CoordinatesType tmp;

};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlNeumannReflectionBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
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


   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );


#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void setBoundaryConditions( const RealType& time,
                               const MeshType& mesh,
                               const IndexType index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& fu );


   CoordinatesType tmp;


};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlNeumannReflectionBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
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


   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );


#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void setBoundaryConditions( const RealType& time,
                               const MeshType& mesh,
                               const IndexType index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& fu );

   private:

   CoordinatesType tmp;


};

#include <operators/tnlNeumannReflectionBoundaryConditions_impl.h>

#endif	/* TNLNEUMANNREFLECTIONBOUNDARYCONDITIONS_H */
