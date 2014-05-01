#ifndef TNLANALYTICSOLUTION_H
#define	TNLANALYTICSOLUTION_H

template<typename Mesh>
class AnalyticSolution
{
   
};

template<typename Real, typename Device, typename Index>
class AnalyticSolution<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:  
   
   typedef tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   
   template< typename TimeFunction, typename AnalyticSpaceFunction>
   void computeAnalyticSolution(const MeshType& mesh, const RealType& time, SharedVector& output, 
                const TimeFunction timeFunction, const AnalyticSpaceFunction analyticSpaceFunction);
   
   template< typename TimeFunction, typename AnalyticSpaceFunction>
   void computeLaplace(const MeshType& mesh, const RealType& time, DofVectorType& output,
                const TimeFunction timeFunction, const AnalyticSpaceFunction analyticSpaceFunction);
};

template<typename Real, typename Device, typename Index>
class AnalyticSolution<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:  
   
   typedef tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   
   template< typename TimeFunction, typename AnalyticSpaceFunction>
   void computeAnalyticSolution(const MeshType& mesh, const RealType& time, SharedVector& output,
                const TimeFunction timeFunction, const AnalyticSpaceFunction analyticSpaceFunction);

   template< typename TimeFunction, typename AnalyticSpaceFunction>
   void computeLaplace(const MeshType& mesh, const RealType& time, DofVectorType& output,
                const TimeFunction timeFunction, const AnalyticSpaceFunction analyticSpaceFunction);
};

template<typename Real, typename Device, typename Index>
class AnalyticSolution<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public:  
   
   typedef tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   
   template< typename TimeFunction, typename AnalyticSpaceFunction>
   void computeAnalyticSolution(const MeshType& mesh, const RealType& time, SharedVector& output,
                const TimeFunction timeFunction, const AnalyticSpaceFunction analyticSpaceFunction);

   template< typename TimeFunction, typename AnalyticSpaceFunction>
   void computeLaplace(const MeshType& mesh, const RealType& time, DofVectorType& output,
                const TimeFunction timeFunction, const AnalyticSpaceFunction analyticSpaceFunction);
};

#include "tnlAnalyticSolution_impl.h"

#endif	/* TNLANALYTICSOLUTION_H */
