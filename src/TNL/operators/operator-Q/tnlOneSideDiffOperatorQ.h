/***************************************************************************
                          tnlOneSidedDiffOperatorQ.h  -  description
                             -------------------
    begin                : Jan 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/vectors/tnlVector.h>
#include <TNL/core/vectors/tnlSharedVector.h>
#include <TNL/mesh/tnlGrid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType > 
class tnlOneSideDiffOperatorQ
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const;
      
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real getValueStriped( const MeshFunction& u,
                         const MeshEntity& entity,   
                         const Real& time = 0.0 ) const;
          
   void setEps(const Real& eps);
      
   private:
   
   RealType eps, epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType(); 
      
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const;

   
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real getValueStriped( const MeshFunction& u,
                         const MeshEntity& entity,          
                         const Real& time = 0.0 ) const;
        
   void setEps( const Real& eps );
   
   private:
   
   RealType eps, epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType();

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const;
   
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real getValueStriped( const MeshFunction& u,
                         const MeshEntity& entity,          
                         const Real& time ) const;
        
   void setEps(const Real& eps);
   
   private:
   
   RealType eps, epsSquare;
};

} // namespace TNL

#include <TNL/operators/operator-Q/tnlOneSideDiffOperatorQ_impl.h>
