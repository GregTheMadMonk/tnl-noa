#ifndef TNLNONLINEARDIFFUSION_H
#define	TNLNONLINEARDIFFUSION_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename NonlinearDiffusionOperator,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlNonlinearDiffusion
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
class tnlNonlinearDiffusion< tnlGrid< 1,MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >
{
   public: 
   
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef NonlinearDiffusionOperator NonlinearDiffusionOperatorType;
      typedef typename MeshType::template MeshEntity< MeshType::meshDimensions > CellType;

      static tnlString getType();

      template< typename MeshEntity,
                typename Vector >
      __cuda_callable__
      Real getValue( const MeshEntity& entity,
                     const Vector& u,
                     const RealType& time) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity,
                typename MeshFunction,
                typename Vector,
                typename Matrix >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const;

   public:
       
      NonlinearDiffusionOperator nonlinearDiffusionOperator;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
class tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >
{
   public: 
   
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef NonlinearDiffusionOperator NonlinearDiffusionOperatorType;


      static tnlString getType();

      template< typename MeshEntity,
                typename Vector >
      __cuda_callable__
      Real getValue( const MeshEntity& entity,
                     const Vector& u,
                     const RealType& time) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity,
                typename MeshFunction,
                typename Vector,
                typename Matrix >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const;
   
   public:
       
      NonlinearDiffusionOperator nonlinearDiffusionOperator;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
class tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >
{
   public: 
   
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef NonlinearDiffusionOperator NonlinearDiffusionOperatorType;

      static tnlString getType();

      template< typename MeshEntity,
                typename MeshFunction >
      __cuda_callable__
      Real getValue( const MeshEntity& entity,
                     const MeshFunction& u,
                     const RealType& time) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity,
                typename MeshFunction,
                typename Vector,                
                typename Matrix >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const;
     
   public:
       
      NonlinearDiffusionOperator nonlinearDiffusionOperator;
};


#include "tnlNonlinearDiffusion_impl.h"


#endif	/* TNLNONLINEARDIFFUSION_H */
