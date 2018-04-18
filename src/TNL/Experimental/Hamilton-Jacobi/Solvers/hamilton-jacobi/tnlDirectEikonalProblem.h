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
#include <TNL/Functions/MeshFunction.h>
#include <TNL/SharedPointer.h>
#include "tnlFastSweepingMethod.h"

template< typename Mesh,
          typename Anisotropy,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlDirectEikonalProblem
   : public Problems::PDEProblem< Mesh,
                                  Real,
                                  typename Mesh::DeviceType,
                                  Index  >
{
   public:
   
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef Problems::PDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      using AnisotropyType = Anisotropy;
      using AnisotropyPointer = SharedPointer< AnisotropyType, DeviceType >;
      using MeshFunctionPointer = SharedPointer< MeshFunctionType >;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;
      using MeshPointer = SharedPointer< MeshType >;
      using DofVectorPointer = SharedPointer< DofVectorType >;
      using MeshDependentDataPointer = SharedPointer< MeshDependentDataType >;
      
      static constexpr bool isTimeDependent() { return false; };

      static String getTypeStatic();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;
      
      bool writeEpilog( Logger& logger );


      bool setup( const MeshPointer& mesh,
                  const Config::ParameterContainer& parameters,
                  const String& prefix );

      IndexType getDofs( const MeshPointer& mesh ) const;

      void bindDofs( const MeshPointer& mesh,
                     const DofVectorPointer& dofs );
      
      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& mesh,
                                DofVectorPointer& dofs,
                                MeshDependentDataPointer& meshdependentData );

      bool solve( const MeshPointer& mesh,
                  DofVectorPointer& dosf );


      protected:
         
         MeshFunctionPointer u;
         
         MeshFunctionPointer initialData;
         
         AnisotropyPointer anisotropy;

};

#include "tnlDirectEikonalProblem_impl.h"
