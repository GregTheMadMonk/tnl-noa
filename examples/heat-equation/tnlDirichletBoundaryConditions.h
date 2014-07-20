#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_H
#define	TNLDIRICHLETBOUNDARYCONDITIONS_H

#include <core/vectors/tnlStaticVector.h>

template< typename Mesh,
          typename Real, // = typename Mesh::RealType,
          typename Index > //= typename Mesh::IndexType >
class tnlDirichletBoundaryConditions
{
   
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlDirichletBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >
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
            
   template< typename AnalyticSpaceFunction,
             typename TimeFunction,
             typename Vector >
   void applyBoundaryConditions( const MeshType& mesh,
                                 Vector& u,
                                 const RealType& time,
                                 TimeFunction& timeFunction,
                                 AnalyticSpaceFunction& analyticSpaceFunction );
   
   template< typename AnalyticSpaceFunction,
             typename TimeFunction,
             typename Vector >
   void applyBoundaryTimeDerivation( const MeshType& mesh,
                                     Vector& u,
                                     const RealType& time,
                                     TimeFunction& timeFunction,
                                     AnalyticSpaceFunction& analyticSpaceFunction );
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlDirichletBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
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
            
   template< typename AnalyticSpaceFunction,
             typename TimeFunction,
             typename Vector >
   void applyBoundaryConditions( const MeshType& mesh,
                                 Vector& u,
                                 const RealType& time,
                                 TimeFunction& timeFunction,
                                 AnalyticSpaceFunction& analyticSpaceFunction );

   template< typename AnalyticSpaceFunction,
             typename TimeFunction,
             typename Vector >
   void applyBoundaryTimeDerivation( const MeshType& mesh,
                                     Vector& u,
                                     const RealType& time,
                                     TimeFunction& timeFunction,
                                     AnalyticSpaceFunction& analyticSpaceFunction );
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlDirichletBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
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
            
   template< typename AnalyticSpaceFunction,
             typename TimeFunction,
             typename Vector >
   void applyBoundaryConditions( const MeshType& mesh,
                                 Vector& u,
                                 const RealType& time,
                                 TimeFunction& timeFunction,
                                 AnalyticSpaceFunction& analyticSpaceFunction );
   
   template< typename AnalyticSpaceFunction,
             typename TimeFunction,
             typename Vector >
   void applyBoundaryTimeDerivation( const MeshType& mesh,
                                     Vector& u,
                                     const RealType& time,
                                     TimeFunction& timeFunction,
                                     AnalyticSpaceFunction& analyticSpaceFunction );
};

#include "tnlDirichletBoundaryConditions_impl.h"

#endif	/* TNLDIRICHLETBOUNDARYCONDITIONS_H */
