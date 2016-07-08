#pragma once

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

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType operator()( const MeshFunction& u,
                              const EntityType& entity,   
                              const RealType& time = 0 ) const;

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


   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType operator()( const MeshFunction& u,
                              const EntityType& entity,   
                              const RealType& time = 0 ) const;


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


   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType operator()( const MeshFunction& u,
                              const EntityType& entity,   
                              const RealType& time = 0 ) const;

   private:

   CoordinatesType tmp;


};

#include <operators/tnlNeumannReflectionBoundaryConditions_impl.h>

#endif	/* TNLNEUMANNREFLECTIONBOUNDARYCONDITIONS_H */
