#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_H
#define	TNLDIRICHLETBOUNDARYCONDITIONS_H

#include <core/vectors/tnlStaticVector.h>
#include <config/tnlParameterContainer.h>
#include <functions/tnlConstantFunction.h>
#include <core/vectors/tnlSharedVector.h>

template< typename Mesh,
          typename Function = tnlConstantFunction< Mesh::Dimensions,
                                                   typename Mesh::RealType >,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlDirichletBoundaryConditions
{
   
};

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >
{
   public:
   
   typedef tnlGrid< Dimensions, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< Dimensions, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
            
   bool setup( const tnlParameterContainer& parameters,
               const tnlString& prefix = "" );

   void setFunction( const Function& function );

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

#ifdef HAVE_CUDA
   __device__ __host__
#endif
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const CoordinatesType& coordinates,
                               DofVectorType& u,
                               DofVectorType& b,
                               IndexType* columns,
                               RealType* values,
                               IndexType& rowLength ) const;

   protected:

   Function function;
};

#include <implementation/operators/tnlDirichletBoundaryConditions_impl.h>

#endif	/* TNLDIRICHLETBOUNDARYCONDITIONS_H */
