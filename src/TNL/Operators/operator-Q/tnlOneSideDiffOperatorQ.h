/***************************************************************************
                          tnlOneSidedDiffOperatorQ.h  -  description
                             -------------------
    begin                : Jan 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Operators {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType > 
class tnlOneSideDiffOperatorQ
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static String getType();

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
class tnlOneSideDiffOperatorQ< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static String getType(); 
      
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
class tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static String getType();

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

} // namespace Operators
} // namespace TNL

#include <TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ_impl.h>
