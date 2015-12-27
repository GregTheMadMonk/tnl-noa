#ifndef TNLFINITEVOLUMENONLINEAROPERATOR_H
#define	TNLFINITEVOLUMENONLINEAROPERATOR_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename NonlinearDiffusionOperator,
	  typename OperatorQ,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlFiniteVolumeNonlinearOperator
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
class tnlFiniteVolumeNonlinearOperator< tnlGrid< 1,MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef OperatorQ OperatorQType;

   static tnlString getType();
   
   template< typename Vector,
             typename MeshEntity >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
                  const IndexType cellIndex,
                  const MeshEntity& entity,
                  const Vector& u,
                  const RealType& time) const;
   
   template< typename MeshEntity >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const MeshEntity& entity ) const;

   template< typename Vector,
             typename MeshEntity,
             typename MatrixRow >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               Vector& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
   
   public:
   
   OperatorQ operatorQ;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
class tnlFiniteVolumeNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef OperatorQ OperatorQType;
   

   static tnlString getType();
   
   template< typename Vector,
             typename MeshEntity >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
                  const IndexType cellIndex,
                  const MeshEntity& entity,
                  const Vector& u,
                  const RealType& time) const;
   
   
   template< typename MeshEntity >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const MeshEntity& entity ) const;

   template< typename Vector,
             typename MeshEntity,
             typename MatrixRow >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               Vector& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
   
   public:
   
   OperatorQ operatorQ;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
class tnlFiniteVolumeNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef OperatorQ OperatorQType;

   static tnlString getType();
   
   template< typename Vector >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
                  const IndexType cellIndex,
                  const MeshEntity& entity,
                  const Vector& u,
                  const RealType& time) const;
   
   template< typename MeshEntity >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const MeshEntity& entity ) const;

   template< typename Vector,
             typename MeshEntity,
             typename MatrixRow >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               Vector& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
   
   public:
   
   OperatorQ operatorQ;
};


#include "tnlFiniteVolumeNonlinearOperator_impl.h"


#endif	/* TNLFINITEVOLUMENONLINEAROPERATOR_H */
