#ifndef TNLFINITEVOLUMEOPERATORQ_H
#define	TNLFINITEVOLUMEOPERATORQ_H

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          int Precomputation = 0 > 
class tnlFiniteVolumeOperatorQ
{

};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, 0 >
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

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time ) 
   {}
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
          
   bool setEps(const Real& eps);
      
   private:
   
      template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
      __cuda_callable__
      Real 
      boundaryDerivative( 
         const MeshType& mesh,
         const MeshEntity& entity,
         const Vector& u,
         const Real& time,
         const IndexType& dx = 0, 
         const IndexType& dy = 0,
         const IndexType& dz = 0 ) const;

      RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, 0 >
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

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time )
   {}   
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real getValue( 
      const MeshType& mesh,
      const MeshEntity& entity,
      const Vector& u,
      const Real& time,
      const IndexType& dx = 0, 
      const IndexType& dy = 0,
      const IndexType& dz = 0 ) const;
        
   bool setEps(const Real& eps);
   
   private:

   template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
   __cuda_callable__
   Real boundaryDerivative( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
   
   RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index, 0 >
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

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time )
   {}
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
        
   bool setEps(const Real& eps);
   
   private:

   template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
   __cuda_callable__
   Real boundaryDerivative( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
   
   RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, 1 >
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

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time );
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
   
   bool setEps(const Real& eps);
   
   private:
   
      template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
      __cuda_callable__
      Real boundaryDerivative( const MeshType& mesh,
             const MeshEntity& entity,
             const Vector& u,
             const Real& time,
             const IndexType& dx = 0, 
             const IndexType& dy = 0,
             const IndexType& dz = 0 ) const;    

      tnlSharedVector< RealType, DeviceType, IndexType > u;
      tnlVector< RealType, DeviceType, IndexType> q;
      RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index, 1 >
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

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time ); 
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
          
   bool setEps(const Real& eps);
   
   private:
   
   template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
   __cuda_callable__
   Real boundaryDerivative( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
       
   tnlSharedVector< RealType, DeviceType, IndexType > u;
   tnlVector< RealType, DeviceType, IndexType> q;
   RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index, 1 >
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

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time );
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
          
   bool setEps(const Real& eps);
   
   private:
   
      template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
      __cuda_callable__
      Real boundaryDerivative( const MeshType& mesh,
             const IndexType cellIndex,
             const MeshEntity& entity,
             const Vector& u,
             const Real& time,
             const IndexType& dx = 0, 
             const IndexType& dy = 0,
             const IndexType& dz = 0 ) const;

      tnlSharedVector< RealType, DeviceType, IndexType > u;
      tnlVector< RealType, DeviceType, IndexType> q;
      RealType eps;
};

#include <operators/operator-Q/tnlFiniteVolumeOperatorQ_impl.h>


#endif	/* TNLFINITEVOLUMEOPERATORQ_H */
