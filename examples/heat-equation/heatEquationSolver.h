/***************************************************************************
                          heatEquationSolver.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef HEATEQUATIONSOLVER_H_
#define HEATEQUATIONSOLVER_H_

#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include "heatEquationSolver.h"

#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlDenseMatrix.h>


template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
class heatEquationSolver
{
   public:

   typedef typename DifferentialOperator::RealType RealType;
   typedef typename Mesh::DeviceType DeviceType;
   typedef typename DifferentialOperator::IndexType IndexType;
   typedef Mesh MeshType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlSlicedEllpackMatrix< RealType, DeviceType, IndexType > MatrixType;
   typedef typename MatrixType::RowLengthsVector RowLengthsVectorType;

   static tnlString getTypeStatic();

   tnlString getPrologHeader() const;

   void writeProlog( tnlLogger& logger,
                     const tnlParameterContainer& parameters ) const;

   bool setup( const tnlParameterContainer& parameters );

   bool setInitialCondition( const tnlParameterContainer& parameters,
                             const MeshType& mesh,
                             DofVectorType& dofs,
                             DofVectorType& auxDofs );

   bool setupLinearSystem( const MeshType& mesh,
                           MatrixType& matrix );

   bool makeSnapshot( const RealType& time,
                      const IndexType& step,
                      const MeshType& mesh,
                      DofVectorType& dofs,
                      DofVectorType& auxDofs );

   IndexType getDofs( const MeshType& mesh ) const;

   IndexType getAuxiliaryDofs( const MeshType& mesh ) const;

   void bindDofs( const MeshType& mesh,
                  DofVectorType& dofs );

   void bindAuxiliaryDofs( const MeshType& mesh,
                           DofVectorType& auxiliaryDofs );

   bool preIterate( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    DofVectorType& dofs,
                    DofVectorType& auxDofs );

   void getExplicitRHS( const RealType& time,
                        const RealType& tau,
                        const MeshType& mesh,
                        DofVectorType& _u,
                        DofVectorType& _fu );


   void assemblyLinearSystem( const RealType& time,
                              const RealType& tau,
                              const MeshType& mesh,
                              DofVectorType& dofs,
                              DofVectorType& auxDofs,
                              MatrixType& matrix,
                              DofVectorType& rightHandSide );

   bool postIterate( const RealType& time,
                     const RealType& tau,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     DofVectorType& auxDofs );


   tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();
   
   protected:

   tnlSharedVector< RealType, DeviceType, IndexType > solution;

   DifferentialOperator differentialOperator;

   BoundaryCondition boundaryCondition;

   RightHandSide rightHandSide;
};

#include "heatEquationSolver_impl.h"

#endif /* HEATEQUATIONSOLVER_H_ */
