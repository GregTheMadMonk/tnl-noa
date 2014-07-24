
#ifndef TNLZERORIGHTHANDSIDE_H
#define	TNLZERORIGHTHANDSIDE_H

template<typename Mesh>
class tnlZeroRightHandSide
{
   
};

template<typename Real, typename Device, typename Index>
class tnlZeroRightHandSide<tnlGrid<1,Real,Device,Index>> {
   
   public:
   
   typedef tnlGrid<1,Real,Device,Index> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction>
   void applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
                       TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
};

template<typename Real, typename Device, typename Index>
class tnlZeroRightHandSide<tnlGrid<2,Real,Device,Index>> {
   
   public:
   
   typedef tnlGrid<2,Real,Device,Index> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction>
   void applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
                       TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
};

template<typename Real, typename Device, typename Index>
class tnlZeroRightHandSide<tnlGrid<3,Real,Device,Index>> {
   
   public:
   
   typedef tnlGrid<3,Real,Device,Index> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction>
   void applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
                       TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
};

#include "tnlZeroRightHandSide_impl.h"

#endif	/* TNLZERORIGHTHANDSIDE_H */

