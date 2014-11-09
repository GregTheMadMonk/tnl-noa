/***************************************************************************
                          hamiltonJacobiProblemSolver.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef HAMILTONJACOBIPROBLEMSOLVER_H_
#define HAMILTONJACOBIPROBLEMSOLVER_H_

#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <matrices/tnlCSRMatrix.h>

template< typename Mesh,
		  typename DifferentialOperator,
		  typename BoundaryCondition,
		  typename RightHandSide>
class hamiltonJacobiProblemSolver
{

   public:

   typedef typename DifferentialOperator::RealType RealType;
   typedef typename Mesh::DeviceType DeviceType;
   typedef typename DifferentialOperator::IndexType IndexType;
   typedef Mesh MeshType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlCSRMatrix< RealType, DeviceType, IndexType > MatrixType;
   typedef typename MatrixType::RowLengthsVector RowLengthsVectorType;

   static tnlString getTypeStatic();

   tnlString getPrologHeader() const;

   void writeProlog( tnlLogger& logger,
                     const tnlParameterContainer& parameters ) const;

   bool setup( const tnlParameterContainer& parameters );

   bool setInitialCondition( const tnlParameterContainer& parameters,
                             const MeshType& mesh,
                             DofVectorType& dofs );

   bool makeSnapshot( const RealType& time,
                      const IndexType& step,
                      const MeshType& mesh );

   IndexType getDofs( const MeshType& mesh ) const;

   IndexType getAuxiliaryDofs( const MeshType& mesh ) const;

   void bindDofs( const MeshType& mesh,
                  DofVectorType& dofs );

   void bindAuxiliaryDofs( const MeshType& mesh,
                           DofVectorType& auxiliaryDofs );

   void getExplicitRHS( const RealType& time,
                        const RealType& tau,
                        const MeshType& mesh,
                        DofVectorType& _u,
                        DofVectorType& _fu );

   bool preIterate( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    DofVectorType& u );

   bool postIterate( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    DofVectorType& u );


   tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();


   protected:

   tnlSharedVector< RealType, DeviceType, IndexType > solution;

   tnlExplicitUpdater< Mesh, DofVectorType, DifferentialOperator, BoundaryCondition, RightHandSide  > explicitUpdater;

   DifferentialOperator differentialOperator;

   BoundaryCondition boundaryCondition;

   RightHandSide rightHandSide;

   bool schemeTest;
   bool tested;
};


#include "hamiltonJacobiProblemSolver_impl.h"



#endif /* HAMILTONJACOBIPROBLEMSOLVER_H_ */
