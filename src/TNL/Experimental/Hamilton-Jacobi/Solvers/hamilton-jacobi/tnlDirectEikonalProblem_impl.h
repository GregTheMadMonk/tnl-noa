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
String
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
getTypeStatic()
{
   
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
String
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
getPrologHeader() const
{
   return String( "Direct eikonal solver" );
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
void
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
writeProlog( tnlLogger& logger,
             const Config::ParameterContainer& parameters ) const
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
   return true;
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
setup( const Config::ParameterContainer& parameters )
{
   return true;
}

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
Index
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
getDofs( const MeshType& mesh ) const
{
   return mesh.template getEntitiesCount< typename MeshType::Cell >();
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

template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
setInitialData( const Config::ParameterContainer& parameters,
                const MeshType& mesh,
                DofVectorType& dofs,
                MeshDependentDataType& meshdependentData )
{
   String inputFile = parameters.getParameter< String >( "input-file" );
   this->initialData.setMesh( mesh );
   if( !this->initialData.boundLoad( inputFile ) )
      return false;
   return true;
}


template< typename Mesh,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Anisotropy, Real, Index >::
solve( const MeshType& mesh,
       DofVectorType& dofs )
{
   tnlFastSweepingMethod< MeshType, AnisotropyType > fsm;
   fsm.solve( mesh, anisotropy, initialData );
}
