#ifndef TNLANALYTICSOLUTION_H
#define	TNLANALYTICSOLUTION_H

template< typename Mesh,
          typename Real,
          typename Index >
class AnalyticSolution
{
   
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class AnalyticSolution< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:  
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   //typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   //typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   
   template< typename TimeFunction,
             typename AnalyticSpaceFunction,
             typename Vector >
   void computeAnalyticSolution( const MeshType& mesh,
                                 const RealType& time,
                                 Vector& output,
                                 const TimeFunction timeFunction,
                                 const AnalyticSpaceFunction analyticSpaceFunction );
   
   template< typename TimeFunction,
             typename AnalyticSpaceFunction,
             typename Vector >
   void computeLaplace( const MeshType& mesh,
                        const RealType& time,
                        Vector& output,
                        const TimeFunction timeFunction,
                        const AnalyticSpaceFunction analyticSpaceFunction );
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class AnalyticSolution< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:  
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   //typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   //typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   
   template< typename TimeFunction,
             typename AnalyticSpaceFunction,
             typename Vector >
   void computeAnalyticSolution( const MeshType& mesh,
                                 const RealType& time,
                                 Vector& output,
                                 const TimeFunction timeFunction,
                                 const AnalyticSpaceFunction analyticSpaceFunction );

   template< typename TimeFunction,
             typename AnalyticSpaceFunction,
             typename Vector >
   void computeLaplace( const MeshType& mesh,
                        const RealType& time,
                        Vector& output,
                        const TimeFunction timeFunction,
                        const AnalyticSpaceFunction analyticSpaceFunction );
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class AnalyticSolution< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:  
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   
   template< typename TimeFunction,
             typename AnalyticSpaceFunction,
             typename Vector >
   void computeAnalyticSolution( const MeshType& mesh,
                                 const RealType& time,
                                 Vector& output,
                                 const TimeFunction timeFunction,
                                 const AnalyticSpaceFunction analyticSpaceFunction );

   template< typename TimeFunction,
             typename AnalyticSpaceFunction,
             typename Vector >
   void computeLaplace( const MeshType& mesh,
                        const RealType& time,
                        Vector& output,
                        const TimeFunction timeFunction,
                        const AnalyticSpaceFunction analyticSpaceFunction );
};

#include "tnlAnalyticSolution_impl.h"

#endif	/* TNLANALYTICSOLUTION_H */
