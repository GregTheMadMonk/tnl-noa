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
#include <TNL/Solvers/PDE/TimeDependentPDESolver.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename Communicator,
          typename Real = typename Mesh::RealType,
          typename Device = typename Mesh::DeviceType,
          typename Index = typename Mesh::GlobalIndexType >
class PDEProblem : public Problem< Real, Device, Index >
{
   public:

      using BaseType = Problem< Real, Device, Index >;
      using typename BaseType::RealType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;

      using MeshType = Mesh;
      using MeshPointer = SharedPointer< MeshType, DeviceType >;
      using DistributedMeshType = Meshes::DistributedMeshes::DistributedMesh< MeshType >;
      using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
      using DofVectorType = Containers::Vector< RealType, DeviceType, IndexType>;
      using DofVectorPointer = SharedPointer< DofVectorType, DeviceType >;
      using MatrixType = Matrices::SlicedEllpack< RealType, DeviceType, IndexType >;
      using CommunicatorType = Communicator;
      
      using MeshDependentDataType = Containers::Vector< RealType, DeviceType, IndexType >;
      using MeshDependentDataPointer = SharedPointer< MeshDependentDataType, DeviceType >;

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
      
      // Width of the subdomain overlaps in case when all of them are the same
      virtual IndexType subdomainOverlapSize();
      
      // Returns default subdomain overlaps i.e. no overlaps on the boundaries, only
      // in the domain interior.
      void getSubdomainOverlaps( const Config::ParameterContainer& parameters,
                                 const String& prefix,
                                 const MeshType& mesh,
                                 SubdomainOverlapsType& lower,
                                 SubdomainOverlapsType& upper );

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
