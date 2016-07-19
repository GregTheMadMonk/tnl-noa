/***************************************************************************
                          tnlPDEProblem.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/problems/tnlProblem.h>
#include <TNL/matrices/tnlCSRMatrix.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Device = typename Mesh::DeviceType,
          typename Index = typename Mesh::IndexType >
class tnlPDEProblem : public tnlProblem< Real, Device, Index >
{
   public:

      typedef tnlProblem< Real, Device, Index > BaseType;
      using typename BaseType::RealType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;

      typedef Mesh MeshType;
      typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
      typedef tnlCSRMatrix< RealType, DeviceType, IndexType > MatrixType;
      typedef tnlVector< RealType, DeviceType, IndexType > MeshDependentDataType;

      /****
       * This means that the time stepper will be set from the command line arguments.
       */
      typedef void TimeStepper;

      static tnlString getTypeStatic();

      tnlString getPrologHeader() const;

      void writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters ) const;
 
      bool writeEpilog( tnlLogger& logger ) const;


      bool setMeshDependentData( const MeshType& mesh,
                                 MeshDependentDataType& meshDependentData );

      void bindMeshDependentData( const MeshType& mesh,
                                  MeshDependentDataType& meshDependentData );

      bool preIterate( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       DofVectorType& dofs,
                       MeshDependentDataType& meshDependentData );
 
      void setExplicitBoundaryConditions( const RealType& time,
                                          const MeshType& mesh,
                                          DofVectorType& dofs,
                                          MeshDependentDataType& meshDependentData );

      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshType& mesh,
                        DofVectorType& dofs,
                        MeshDependentDataType& meshDependentData );

      tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();


};

} //namespace TNL

#include <TNL/problems/tnlPDEProblem_impl.h>
