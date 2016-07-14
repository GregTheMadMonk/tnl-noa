/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlFastSweepingMethod2D_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 10:32 AM
 */

#pragma once

#include "tnlFastSweepingMethod.h"

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
tnlFastSweepingMethod< tnlGrid< 3, Real, Device, Index >, Anisotropy >::
tnlFastSweepingMethod()
: maxIterations( 1 )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
const Index&
tnlFastSweepingMethod< tnlGrid< 3, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
tnlFastSweepingMethod< tnlGrid< 3, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{
   
}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
tnlFastSweepingMethod< tnlGrid< 3, Real, Device, Index >, Anisotropy >::
solve( const MeshType& mesh,
       const AnisotropyType& anisotropy,
       MeshFunctionType& u )
{
   MeshFunctionType aux;
   aux.setMesh( mesh );
   std::cout << "Initiating the interface cells ..." << std::endl;
   BaseType::initInterface( u, aux );
   aux.save( "aux.tnl" );
}

