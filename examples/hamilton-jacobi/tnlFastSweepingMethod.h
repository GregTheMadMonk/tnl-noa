/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlFastSweepingMethod.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 10:04 AM
 */

#pragma once

#include <mesh/tnlGrid.h>
#include <functions/tnlConstantFunction.h>
#include "tnlDirectEikonalMethodsBase.h"

template< typename Mesh,
          typename Anisotropy = tnlConstantFunction< Mesh::getMeshDimensions(), typename Mesh::RealType > >
class tnlFastSweepingMethod
{   
};

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
class tnlFastSweepingMethod< tnlGrid< 1, Real, Device, Index >, Anisotropy >
   : public tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > >
{
   static_assert(  std::is_same< Device, tnlHost >::value, "The fast sweeping method works only on CPU." );
   
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef Anisotropy AnisotropyType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      typedef tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > > BaseType;
      
      tnlFastSweepingMethod();
      
      const IndexType& getMaxIterations() const;
      
      void setMaxIterations( const IndexType& maxIterations );
      
      void solve( const MeshType& mesh,
                  const AnisotropyType& anisotropy,
                  MeshFunctionType& u );
      
      
   protected:
      
      const IndexType maxIterations;
};

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
class tnlFastSweepingMethod< tnlGrid< 2, Real, Device, Index >, Anisotropy >
   : public tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > >
{
   static_assert(  std::is_same< Device, tnlHost >::value, "The fast sweeping method works only on CPU." );
   
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef Anisotropy AnisotropyType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      typedef tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > > BaseType;            
      
      tnlFastSweepingMethod();
      
      const IndexType& getMaxIterations() const;
      
      void setMaxIterations( const IndexType& maxIterations );
      
      void solve( const MeshType& mesh,
                  const AnisotropyType& anisotropy,
                  MeshFunctionType& u );
      
      
   protected:
      
      const IndexType maxIterations;
};

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
class tnlFastSweepingMethod< tnlGrid< 3, Real, Device, Index >, Anisotropy >
   : public tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > >
{
   static_assert(  std::is_same< Device, tnlHost >::value, "The fast sweeping method works only on CPU." );
   
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef Anisotropy AnisotropyType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      typedef tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > > BaseType;            
      
      tnlFastSweepingMethod();
      
      const IndexType& getMaxIterations() const;
      
      void setMaxIterations( const IndexType& maxIterations );
      
      void solve( const MeshType& mesh,
                  const AnisotropyType& anisotropy,
                  MeshFunctionType& u );
      
      
   protected:
      
      const IndexType maxIterations;
};



#include "tnlFastSweepingMethod1D_impl.h"
#include "tnlFastSweepingMethod2D_impl.h"
#include "tnlFastSweepingMethod3D_impl.h"
