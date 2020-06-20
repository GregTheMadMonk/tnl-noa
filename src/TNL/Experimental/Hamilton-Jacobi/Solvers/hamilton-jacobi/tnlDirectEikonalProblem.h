/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlFastSweepingMethod.h
 * Author: oberhuber
 *
 * Created on July 13, 2016, 1:19 PM
 */

#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Pointers/SharedPointer.h>
#include "tnlFastSweepingMethod.h"

#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>

template< typename Mesh,
          typename Communicator,
          typename Anisotropy,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlDirectEikonalProblem
   : public Problems::PDEProblem< Mesh,
                                  Communicator,
                                  Real,
                                  typename Mesh::DeviceType,
                                  Index  >
{
   public:
   
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
      typedef Problems::PDEProblem< Mesh, Communicator, RealType, DeviceType, IndexType > BaseType;
      using AnisotropyType = Anisotropy;
      using AnisotropyPointer = Pointers::SharedPointer< AnisotropyType, DeviceType >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using MeshPointer = Pointers::SharedPointer< MeshType >;
      using DofVectorPointer = Pointers::SharedPointer< DofVectorType >;
      
      typedef Communicator CommunicatorType;
      
      static constexpr bool isTimeDependent() { return false; };

      static String getType();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;
      
      bool writeEpilog( Logger& logger );


      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofs );
      
      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofs );
      
      bool makeSnapshot( );

      bool solve( DofVectorPointer& dosf );


      protected:
         
         using DistributedMeshSynchronizerType = Meshes::DistributedMeshes::DistributedMeshSynchronizer< MeshFunctionType >;
         DistributedMeshSynchronizerType synchronizer;

         MeshFunctionPointer u;
         
         MeshFunctionPointer initialData;
         
         AnisotropyPointer anisotropy;

         Meshes::DistributedMeshes::DistrGridIOTypes distributedIOType;
         
   };

#include "tnlDirectEikonalProblem_impl.h"
