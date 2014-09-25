#ifndef TNLRIGHTHANDSIDE_H
#define	TNLRIGHTHANDSIDE_H

#include <core/vectors/tnlStaticVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real, // = typename Mesh::RealType,
          typename Index > //= typename Mesh::IndexType >
class tnlRightHandSide
{
   
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlRightHandSide< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   
   public:
   
   typedef tnlGrid< 1,MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   
   template< typename AnalyticSpaceFunction,
             typename TimeFunction >
   void applyRHSValues( const MeshType& mesh,
                        const RealType& time,
                        DofVectorType& _fu,
                        TimeFunction& timeFunction,
                        AnalyticSpaceFunction& analyticSpaceFunction );
   
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlRightHandSide< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   
   public:
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   
   template< typename AnalyticSpaceFunction,
             typename TimeFunction >
   void applyRHSValues( const MeshType& mesh,
                        const RealType& time,
                        DofVectorType& _fu,
                        TimeFunction& timeFunction,
                        AnalyticSpaceFunction& analyticSpaceFunction );
   
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlRightHandSide< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index  >
{
   
   public:
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   
   template< typename AnalyticSpaceFunction,
             typename TimeFunction >
   void applyRHSValues( const MeshType& mesh,
                        const RealType& time,
                        DofVectorType& _fu,
                        TimeFunction& timeFunction,
                        AnalyticSpaceFunction& analyticSpaceFunction );
   
};

#include "tnlRightHandSide_impl.h"

#endif	/* TNLRIGHTHANDSIDE_H */

