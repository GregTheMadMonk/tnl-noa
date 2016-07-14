/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlFastSweepingMethod_impl.h
 * Author: oberhuber
 *
 * Created on July 13, 2016, 1:46 PM
 */

#pragma once

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
tnlString
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
getTypeStatic()
{
   
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
tnlString
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
getPrologHeader() const
{
   
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
void
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
writeProlog( tnlLogger& logger,
             const tnlParameterContainer& parameters ) const
{
   
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
writeEpilog( tnlLogger& logger )
{
   
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
setup( const tnlParameterContainer& parameters )
{
   
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
Index
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
getDofs( const MeshType& mesh ) const
{
   return mesh.getEntitiesCount();
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
void
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
bindDofs( const MeshType& mesh,
          const DofVectorType& dofs )
{
   this->u.bind( mesh, dofs );
}

bool
setInitialData( const tnlParameterContainer& parameters,
                const MeshType& mesh,
                DofVectorType& dofs,
                MeshDependentDataType& meshdependentData )
{
   tnlString inputFile = parameters.getParameter< tnlString >( "input-file" );
   this->initialData.setMesh( mesh );
   if( !this->initialData.boundLoad( inputFile ) )
      return false;
   
}


template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
solve( const MeshType& mesh,
       DofVectorType& dosf )
{
   
}