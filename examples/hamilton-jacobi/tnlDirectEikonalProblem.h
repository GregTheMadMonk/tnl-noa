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

#include <problems/tnlPDEProblem.h>
#include <functions/tnlMeshFunction.h>
#include "tnlFastSweepingMethod.h"

template< typename Mesh,
          typename Anisotropy,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlDirectEikonalProblem
   : public tnlPDEProblem< Mesh,
                           TimeIndependentProblem,
                           Real,
                           typename Mesh::DeviceType,
                           Index  >
{
   public:
   
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlPDEProblem< Mesh, TimeIndependentProblem, RealType, DeviceType, IndexType > BaseType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;

      static tnlString getTypeStatic();

      tnlString getPrologHeader() const;

      void writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters ) const;
      
      bool writeEpilog( tnlLogger& logger );


      bool setup( const tnlParameterContainer& parameters );

      IndexType getDofs( const MeshType& mesh ) const;

      void bindDofs( const MeshType& mesh,
                     const DofVectorType& dofs );
      
      bool setInitialData( const tnlParameterContainer& parameters,
                           const MeshType& mesh,
                           DofVectorType& dofs,
                           MeshDependentDataType& meshdependentData );

      bool solve( const MeshType& mesh,
                  DofVectorType& dosf );


      protected:
         
         MeshFunctionType u;
         
         MeshFunctionType initialData;

};

#include "tnlDirectEikonalProblem_impl.h"
