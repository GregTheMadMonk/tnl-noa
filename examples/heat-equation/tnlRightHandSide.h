#ifndef TNLRIGHTHANDSIDE_H
#define	TNLRIGHTHANDSIDE_H

#include <core/vectors/tnlStaticVector.h>

template<typename Mesh>
class tnlRightHandSide
{
   
};

template<typename Real, typename Device, typename Index>
class tnlRightHandSide<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>> {
   
   public:
   
   typedef tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction>
   void applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
                       TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
};

template<typename Real, typename Device, typename Index>
class tnlRightHandSide<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>> {
   
   public:
   
   typedef tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction>
   void applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
                       TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
};

template<typename Real, typename Device, typename Index>
class tnlRightHandSide<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>> {
   
   public:
   
   typedef tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction>
   void applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
                       TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
};

#include "tnlRightHandSide_impl.h"

#endif	/* TNLRIGHTHANDSIDE_H */

