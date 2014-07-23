#ifndef TNLLINEARDIFFUSION_H
#define	TNLLINEARDIFFUSION_H

#include <core/vectors/tnlVector.h>

template< typename Mesh,
          typename Real,// = typename Mesh::RealType,
          typename Index >// = typename Mesh::IndexType >
class tnlLinearDiffusion
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   //typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   template< typename Vector >
   void getExplicitRHS( const MeshType& mesh,
                        const CoordinatesType& coordinates,
                        Vector& _u,
                        Vector& _fu );

   template< typename Vector >
   void getExplicitRHS( const MeshType& mesh,
                        Vector& _u,
                        Vector& _fu );
   
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   //typedef tnlVector< RealType, DeviceType, IndexType > DofVectorType;

   template< typename Vector >
   void getExplicitRHS( const MeshType& mesh,
                        const CoordinatesType& coordinates,
                        Vector& _u,
                        Vector& _fu );

   template< typename Vector >
   void getExplicitRHS( const MeshType& mesh,
                        Vector& _u,
                        Vector& _fu);
   
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   //typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   template< typename Vector >
   void getExplicitRHS( const MeshType& mesh,
                        const CoordinatesType& coordinates,
                        Vector& _u,
                        Vector& _fu );

   template< typename Vector >
   void getExplicitRHS( const MeshType& mesh,
                        Vector& _u,
                        Vector& _fu );
   
};


#include "tnlLinearDiffusion_impl.h"


#endif	/* TNLLINEARDIFFUSION_H */
