#ifndef TNLLINEARDIFFUSION_H
#define	TNLLINEARDIFFUSION_H

template<typename Mesh>
class tnlLinearDiffusion
{
 
};


template<typename Real, typename Device, typename Index>
class tnlLinearDiffusion<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>> 
{
   public: 
   
   typedef tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   void getExplicitRHS( const MeshType& mesh,
                        const CoordinatesType& coordinates,
                        DofVectorType& _u,
                        DofVectorType& _fu
                        );

   void getExplicitRHS( const MeshType& mesh,
                        DofVectorType& _u,
                        DofVectorType& _fu);
   
};


template<typename Real, typename Device, typename Index>
class tnlLinearDiffusion<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public: 
   
   typedef tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   void getExplicitRHS( const MeshType& mesh,
                        const CoordinatesType& coordinates,
                        DofVectorType& _u,
                        DofVectorType& _fu
                        );

   void getExplicitRHS( const MeshType& mesh,
                        DofVectorType& _u,
                        DofVectorType& _fu);
   
};


template<typename Real, typename Device, typename Index>
class tnlLinearDiffusion<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>
{
   public: 
   
   typedef tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry> MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef typename MeshType::RealType RealType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   void getExplicitRHS( const MeshType& mesh,
                        const CoordinatesType& coordinates,
                        DofVectorType& _u,
                        DofVectorType& _fu
                        );

   void getExplicitRHS( const MeshType& mesh,
                        DofVectorType& _u,
                        DofVectorType& _fu);
   
};


#include "tnlLinearDiffusion_impl.h"


#endif	/* TNLLINEARDIFFUSION_H */
