#ifndef TNLNEUMANNBOUNDARYCONDITIONS_H
#define	TNLNEUMANNBOUNDARYCONDITIONS_H

template<typename Mesh>
class tnlNeumannBoundaryConditions  
{
   
};

template<typename Real, typename Device, typename Index>
class tnlNeumannBoundaryConditions<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:
   
   typedef tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef typename MeshType::RealType RealType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
      
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryConditions(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction);

   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction);
};

template<typename Real, typename Device, typename Index>
class tnlNeumannBoundaryConditions<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:
   
   typedef tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef typename MeshType::RealType RealType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
      
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryConditions(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction);
   
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction);
};

template<typename Real, typename Device, typename Index>
class tnlNeumannBoundaryConditions<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:
   
   typedef tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef typename MeshType::RealType RealType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryConditions(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction); 
   
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction);
};

#include "tnlNeumannBoundaryConditions_impl.h"

#endif	/* TNLNEUMANNBOUNDARYCONDITIONS_H */

