/***************************************************************************
                          FastSweepingMethod.h  -  description
                             -------------------
    begin                : Jul 14, 2016
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/SharedPointer.h>
#include "tnlDirectEikonalMethodsBase.h"

template< typename Mesh,
          typename Anisotropy = Functions::Analytic::Constant< Mesh::getMeshDimension(), typename Mesh::RealType > >
class FastSweepingMethod
{   
};

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
class FastSweepingMethod< Meshes::Grid< 1, Real, Device, Index >, Anisotropy >
   : public tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >
{
   //static_assert(  std::is_same< Device, TNL::Devices::Host >::value, "The fast sweeping method works only on CPU." );
   
   public:
      
      typedef Meshes::Grid< 1, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef TNL::Devices::Host DeviceType;
      typedef Index IndexType;
      typedef Anisotropy AnisotropyType;
      typedef tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > > BaseType;
      using MeshPointer = SharedPointer< MeshType >;
      
      using typename BaseType::InterfaceMapType;
      using typename BaseType::MeshFunctionType;
      
      FastSweepingMethod();
      
      const IndexType& getMaxIterations() const;
      
      void setMaxIterations( const IndexType& maxIterations );
      
      void solve( const MeshPointer& mesh,
                  const AnisotropyType& anisotropy,
                  MeshFunctionType& u );
      
      
   protected:
      
      const IndexType maxIterations;
};

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
class FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >
   : public tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >
{
   //static_assert(  std::is_same< Device, TNL::Devices::Host >::value, "The fast sweeping method works only on CPU." );
   
   public:
      
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef TNL::Devices::Host DeviceType;
      typedef Index IndexType;
      typedef Anisotropy AnisotropyType;
      typedef tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > BaseType;
      using MeshPointer = SharedPointer< MeshType >;

      using typename BaseType::InterfaceMapType;
      using typename BaseType::MeshFunctionType;

      FastSweepingMethod();
      
      const IndexType& getMaxIterations() const;
      
      void setMaxIterations( const IndexType& maxIterations );
      
      void solve( const MeshPointer& mesh,
                  const AnisotropyType& anisotropy,
                  MeshFunctionType& u );
      
      
   protected:
      
      const IndexType maxIterations;
};

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
class FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >
   : public tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >
{
   //static_assert(  std::is_same< Device, TNL::Devices::Host >::value, "The fast sweeping method works only on CPU." );
   
   public:
      
      typedef Meshes::Grid< 3, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef TNL::Devices::Host DeviceType;
      typedef Index IndexType;
      typedef Anisotropy AnisotropyType;
      typedef tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > BaseType;
      using MeshPointer = SharedPointer< MeshType >;
      
      using typename BaseType::InterfaceMapType;
      using typename BaseType::MeshFunctionType;
      
      FastSweepingMethod();
      
      const IndexType& getMaxIterations() const;
      
      void setMaxIterations( const IndexType& maxIterations );
      
      void solve( const MeshPointer& mesh,
                  const AnisotropyType& anisotropy,
                  MeshFunctionType& u );
      
      
   protected:
      
      const IndexType maxIterations;
};



#include "tnlFastSweepingMethod1D_impl.h"
#include "tnlFastSweepingMethod2D_impl.h"
#include "tnlFastSweepingMethod3D_impl.h"
