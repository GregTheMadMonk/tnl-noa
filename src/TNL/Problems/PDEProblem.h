/***************************************************************************
                          PDEProblem.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Problems/Problem.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Matrices/SlicedEllpack.h>
#include <TNL/Solvers/PDE/TimeDependentPDESolver.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Device = typename Mesh::DeviceType,
          typename Index = typename Mesh::GlobalIndexType >
class PDEProblem : public Problem< Real, Device, Index >
{
   public:

      typedef Problem< Real, Device, Index > BaseType;
      using typename BaseType::RealType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;

      typedef Mesh MeshType;
      typedef Pointers::SharedPointer<  MeshType, DeviceType > MeshPointer;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef Pointers::SharedPointer<  DofVectorType, DeviceType > DofVectorPointer;
      typedef Matrices::SlicedEllpack< RealType, DeviceType, IndexType > MatrixType;
      typedef Containers::Vector< RealType, DeviceType, IndexType > MeshDependentDataType;
      typedef Pointers::SharedPointer<  MeshDependentDataType, DeviceType > MeshDependentDataPointer;

      static constexpr bool isTimeDependent() { return true; };
      
      /****
       * This means that the time stepper will be set from the command line arguments.
       */
      typedef void TimeStepper;
      
      static String getType();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;
 
      bool writeEpilog( Logger& logger ) const;


      bool setMeshDependentData( const MeshPointer& mesh,
                                 MeshDependentDataPointer& meshDependentData );

      void bindMeshDependentData( const MeshPointer& mesh,
                                  MeshDependentDataPointer& meshDependentData );

      bool preIterate( const RealType& time,
                       const RealType& tau,
                       const MeshPointer& meshPointer,
                       DofVectorPointer& dofs,
                       MeshDependentDataPointer& meshDependentData );
 
      void setExplicitBoundaryConditions( const RealType& time,
                                          const MeshPointer& meshPointer,
                                          DofVectorPointer& dofs,
                                          MeshDependentDataPointer& meshDependentData );

      template< typename Matrix >
      void saveFailedLinearSystem( const Matrix& matrix,
                                   const DofVectorType& dofs,
                                   const DofVectorType& rightHandSide ) const;

      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshPointer& meshPointer,
                        DofVectorPointer& dofs,
                        MeshDependentDataPointer& meshDependentData );

      Solvers::SolverMonitor* getSolverMonitor();
};

} // namespace Problems
} // namespace TNL

#include <TNL/Problems/PDEProblem_impl.h>
