#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_H
#define	TNLDIRICHLETBOUNDARYCONDITIONS_H

#include <core/vectors/tnlStaticVector.h>

template<typename Mesh>
class tnlDirichletBoundaryConditions
{
   
};

template<typename Real, typename Device, typename Index>
class tnlDirichletBoundaryConditions<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:
   
   typedef tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef typename MeshType::RealType RealType; 
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
            
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
class tnlDirichletBoundaryConditions<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:
   
   typedef tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef typename MeshType::RealType RealType; 
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
            
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
class tnlDirichletBoundaryConditions<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:
   
   typedef tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef typename MeshType::RealType RealType; 
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
            
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time, 
           TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction);
   
   template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
   void applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, 
           const RealType& time, TimeFunction& timeFunction, 
           AnalyticSpaceFunction& analyticSpaceFunction);
};

#include "tnlDirichletBoundaryConditions_impl.h"

#endif	/* TNLDIRICHLETBOUNDARYCONDITIONS_H */