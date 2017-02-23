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
#include <TNL/SharedPointer.h>
#include <TNL/Matrices/SlicedEllpack.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Device = typename Mesh::DeviceType,
          typename Index = typename Mesh::IndexType >
class PDEProblem : public Problem< Real, Device, Index >
{
   public:

      typedef Problem< Real, Device, Index > BaseType;
      using typename BaseType::RealType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;

      typedef Mesh MeshType;
      typedef SharedPointer< MeshType, DeviceType > MeshPointer;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
      typedef Matrices::SlicedEllpack< RealType, DeviceType, IndexType > MatrixType;
      typedef Containers::Vector< RealType, DeviceType, IndexType > MeshDependentDataType;
      typedef SharedPointer< MeshDependentDataType, DeviceType > MeshDependentDataPointer;

      /****
       * This means that the time stepper will be set from the command line arguments.
       */
      typedef void TimeStepper;

      static String getTypeStatic();

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
