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
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
String
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
getType()
{
   return String( "DirectEikonalProblem< " + 
                  Mesh::getType() + ", " +
                  Anisotropy::getType() + ", " +
                  Real::getType() + ", " +
                  Index::getType() + " >" );
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
String
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
getPrologHeader() const
{
   return String( "Direct eikonal solver" );
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
void
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
writeProlog( Logger& logger,
             const Config::ParameterContainer& parameters ) const
{
   
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
writeEpilog( Logger& logger )
{
   return true;
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return true;
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
Index
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
getDofs() const
{
   return this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
void
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
bindDofs( const DofVectorPointer& dofs )
{
   this->u->bind( this->getMesh(), dofs );
}

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
  this->bindDofs( dofs );
  String inputFile = parameters.getParameter< String >( "input-file" );
  this->initialData->setMesh( this->getMesh() );
  std::cout<<"setInitialCondition" <<std::endl; 
  if(CommunicatorType::isDistributed())
  {
    std::cout<<"Nodes Distribution: " << u->getMesh().getDistributedMesh()->printProcessDistr() << std::endl;
    if(distributedIOType==Meshes::DistributedMeshes::MpiIO)
      Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::load(inputFile, *u );
    if(distributedIOType==Meshes::DistributedMeshes::LocalCopy)
      Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::LocalCopy> ::load(inputFile, *u );
    u->template synchronize<CommunicatorType>();
  }
  else
  {
      try
      {
          this->initialData->boundLoad( inputFile );
      }
      catch(...)
      {
         std::cerr << "I am not able to load the initial condition from the file " << inputFile << "." << std::endl;
         return false;
      }
  }
  return true;
}


template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real,
          typename Index >
bool
tnlDirectEikonalProblem< Mesh, Communicator, Anisotropy, Real, Index >::
solve( DofVectorPointer& dofs )
{
   FastSweepingMethod< MeshType, AnisotropyType > fsm;
   fsm.solve( this->getMesh(), anisotropy, initialData );
   return true;
}
