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
#include <TNL/FileName.h>

#include "tnlDirectEikonalProblem.h"

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
  String param=parameters.getParameter< String >( "distributed-grid-io-type" );
   if(param=="MpiIO")
        distributedIOType=Meshes::DistributedMeshes::MpiIO;
   if(param=="LocalCopy")
        distributedIOType=Meshes::DistributedMeshes::LocalCopy;
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
  if( CommunicatorType::isDistributed() )
  {
    std::cout<<"Nodes Distribution: " << initialData->getMesh().getDistributedMesh()->printProcessDistr() << std::endl;
    if(distributedIOType==Meshes::DistributedMeshes::MpiIO)
      Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::load(inputFile, *initialData );
    if(distributedIOType==Meshes::DistributedMeshes::LocalCopy)
      Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::LocalCopy> ::load(inputFile, *initialData );
    initialData->template synchronize<CommunicatorType>();
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
makeSnapshot(  )
{
   std::cout << std::endl << "Writing output." << std::endl;

   //this->bindDofs( dofs );

   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setExtension( "tnl" );

   if(CommunicatorType::isDistributed())
   {
      if(distributedIOType==Meshes::DistributedMeshes::MpiIO)
        Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::MpiIO> ::save(fileName.getFileName(), *u );
      if(distributedIOType==Meshes::DistributedMeshes::LocalCopy)
        Meshes::DistributedMeshes::DistributedGridIO<MeshFunctionType,Meshes::DistributedMeshes::LocalCopy> ::save(fileName.getFileName(), *u );
   }
   else
   {
      if( ! this->u->save( fileName.getFileName() ) )
         return false;
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
   FastSweepingMethod< MeshType, Communicator,AnisotropyType > fsm;
   fsm.solve( this->getMesh(), u, anisotropy, initialData );
   
   /*int i = Communicators::MpiCommunicator::GetRank( Communicators::MpiCommunicator::AllGroup );
   const MeshPointer msh = this->getMesh();
   if( i == 0 &&  msh->getMeshDimension() == 2 )
   {
     for( int k = 0; k < 9; k++ ){
       for( int l = 0; l < msh->getDimensions().x(); l++ )
         printf("%.2f\t",(*initialData)[ k * msh->getDimensions().x() + l ] );
       printf("\n");
     }
   }*/
   makeSnapshot();
   return true;
}
