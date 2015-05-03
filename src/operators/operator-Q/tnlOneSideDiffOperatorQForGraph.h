#ifndef TNLONESIDEDIFFOPERATORQFORGRAPH_H
#define	TNLONESIDEDIFFOPERATORQFORGRAPH_H

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          int Precomputation = 0 > 
class tnlOneSideDiffOperatorQForGraph
{

};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQForGraph< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, 0 >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename Vector >
   IndexType bind( Vector& u) 
   { return 0; }

#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   void update( const MeshType& mesh, const RealType& time ) 
   {}
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time )const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQForGraph< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, 0 >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType(); 

   template< typename Vector >
   IndexType bind( Vector& u)
   { return 0; }

#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   void update( const MeshType& mesh, const RealType& time )
   {}   
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time )const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQForGraph< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index, 0 >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename Vector >
   IndexType bind( Vector& u)
   { return 0; }

#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   void update( const MeshType& mesh, const RealType& time )
   {}
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQForGraph< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, 1 >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename Vector >
   Index bind( Vector& u);

#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   void update( const MeshType& mesh, const RealType& time );
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   private:
   
   tnlSharedVector< RealType, DeviceType, IndexType > u;
   tnlVector< RealType, DeviceType, IndexType> q;
   tnlVector< RealType, DeviceType, IndexType> qStriped;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQForGraph< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, 1 >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > DofVectorType;
   
   static tnlString getType(); 

   template< typename Vector >
   Index bind( Vector& u);

#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   void update( const MeshType& mesh, const RealType& time ); 
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time )const;
   
   private:
   
   tnlSharedVector< RealType, DeviceType, IndexType > u;
   tnlVector< RealType, DeviceType, IndexType> q;
   tnlVector< RealType, DeviceType, IndexType> qStriped;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQForGraph< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index, 1 >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   
   static tnlString getType();

   template< typename Vector >
   Index bind( Vector& u);

#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   void update( const MeshType& mesh, const RealType& time );
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const;
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif   
   Real getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time )const;
   
   private:
   
   tnlSharedVector< RealType, DeviceType, IndexType > u;
   tnlVector< RealType, DeviceType, IndexType> q;
   tnlVector< RealType, DeviceType, IndexType> qStriped;
};

#include <operators/operator-Q/tnlOneSideDiffOperatorQForGraph_impl.h>


#endif	/* TNLONESIDEDIFFOPERATORQFORGRAPH_H */
