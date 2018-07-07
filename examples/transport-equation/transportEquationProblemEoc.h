/***************************************************************************
                          transportEquationProblemEoc.h  -  description
                             -------------------
    begin                : Feb 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/SharedPointer.h>
#include "transportEquationProblem.h"

using namespace TNL::Problems;

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
class transportEquationProblemEoc:
public transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator > BaseType;
      typedef SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef SharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef typename DifferentialOperator::VelocityFieldType VelocityFieldType;
      typedef SharedPointer< VelocityFieldType, DeviceType > VelocityFieldPointer;
      

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;
      using typename BaseType::MeshDependentDataPointer; 
      
      //using BaseType::getExplicitUpdate;
      
      static String getType();

      String getPrologHeader() const;

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& mesh,
                                DofVectorPointer& dofs,
                                MeshDependentDataPointer& meshDependentData );

      /*template< typename Matrix >
      bool setupLinearSystem( const MeshPointer& mesh,
                              Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshPointer& mesh,
                         DofVectorPointer& dofs,
                         MeshDependentDataPointer& meshDependentData );

      IndexType getDofs( const MeshPointer& mesh ) const;

      void bindDofs( const MeshPointer& mesh,
                     DofVectorPointer& dofs );

      void getExplicitUpdate( const RealType& time,
                           const RealType& tau,
                           const MeshPointer& mesh,
                           DofVectorPointer& _u,
                           DofVectorPointer& _fu,
                           MeshDependentDataPointer& meshDependentData );*/


   protected:
};

} // namespace TNL

#include "transportEquationProblemEoc_impl.h"

