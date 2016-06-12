/***************************************************************************
                          tnlLinearSystemAssembler.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLLINEARSYSTEMASSEMBLER_H_
#define TNLLINEARSYSTEMASSEMBLER_H_

#include <functions/tnlFunctionAdapter.h>

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix,
          typename DofVector >
class tnlLinearSystemAssemblerTraverserUserData
{
   public:
      typedef Matrix MatrixType;
      typedef typename Matrix::DeviceType DeviceType;
      typedef tnlSharedPointer< Matrix, DeviceType > MatrixPointer;
      typedef tnlSharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef tnlSharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef tnlSharedPointer< RightHandSide, DeviceType > RightHandSidePointer;


      const Real* time;

      const Real* tau;

      const Real* timeDiscretisationCoefficient;

      const DifferentialOperatorPointer differentialOperatorPointer;

      const BoundaryConditionsPointer boundaryConditionsPointer;

      const RightHandSidePointer rightHandSidePointer;
      
      const MeshFunction *u;
      
      DofVector *b;

      MatrixPointer matrixPointer;

      tnlLinearSystemAssemblerTraverserUserData( const Real& time,
                                                 const Real& tau,
                                                 const DifferentialOperatorPointer& differentialOperatorPointer,
                                                 const BoundaryConditionsPointer& boundaryConditionsPointer,
                                                 const RightHandSidePointer& rightHandSidePointer,
                                                 const MeshFunction& u,
                                                 MatrixPointer& matrixPointer,
                                                 DofVector& b )
      : time( &time ),
        tau( &tau ),
        differentialOperatorPointer( differentialOperatorPointer ),
        boundaryConditionsPointer( boundaryConditionsPointer ),
        rightHandSidePointer( rightHandSidePointer ),
        u( &u ),
        b( &b ),
        matrixPointer( matrixPointer )
      {};

   protected:

};


template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename Matrix,
          typename DofVector >
class tnlLinearSystemAssembler
{
   public:
   typedef typename MeshFunction::MeshType MeshType;
   typedef typename MeshFunction::MeshPointer MeshPointer;
   typedef typename MeshFunction::RealType RealType;
   typedef typename MeshFunction::DeviceType DeviceType;
   typedef typename MeshFunction::IndexType IndexType;
   typedef Matrix MatrixType;
   typedef tnlLinearSystemAssemblerTraverserUserData< RealType,
                                                      MeshFunction,                                                      
                                                      DifferentialOperator,
                                                      BoundaryConditions,
                                                      RightHandSide,
                                                      MatrixType,
                                                      DofVector > TraverserUserData;

   typedef tnlSharedPointer< Matrix, DeviceType > MatrixPointer;
   typedef tnlSharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
   typedef tnlSharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
   typedef tnlSharedPointer< RightHandSide, DeviceType > RightHandSidePointer;   
      
   template< typename EntityType >
   void assembly( const RealType& time,
                  const RealType& tau,
                  const MeshPointer& meshPointer,
                  const DifferentialOperatorPointer& differentialOperatorPointer,
                  const BoundaryConditionsPointer& boundaryConditionsPointer,
                  const RightHandSidePointer& rightHandSidePointer,
                  const MeshFunction& u,
                  MatrixPointer& matrixPointer,
                  DofVector& b ) const;

   
      class TraverserBoundaryEntitiesProcessor
   {
      public:
         
         template< typename EntityType >         
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
             ( *userData.b )[ entity.getIndex() ] = 0.0;           
             userData.boundaryConditionsPointer.template getData< DeviceType >().setMatrixElements
               ( *userData.u,
                 entity,
                 *userData.time + *userData.tau,
                 *userData.tau,
                 userData.matrixPointer.template modifyData< DeviceType >(),
                 *userData.b );
         }
   };

   class TraverserInteriorEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.b )[ entity.getIndex() ] = 0.0;            
            userData.differentialOperatorPointer.template getData< DeviceType >().setMatrixElements
               ( *userData.u,
                 entity,
                 *userData.time + *userData.tau,
                 *userData.tau,
                 userData.matrixPointer.template modifyData< DeviceType >(), 
                 *userData.b );
            
            typedef tnlFunctionAdapter< MeshType, RightHandSide > RhsFunctionAdapter;
            typedef tnlFunctionAdapter< MeshType, MeshFunction > MeshFunctionAdapter;
            const RealType& rhs = RhsFunctionAdapter::getValue
               ( userData.rightHandSidePointer.template getData< DeviceType >(),
                 entity,
                 *userData.time );
            TimeDiscretisation::applyTimeDiscretisation( userData.matrixPointer.template modifyData< DeviceType >(),
                                                         ( *userData.b )[ entity.getIndex() ],
                                                         entity.getIndex(),
                                                         MeshFunctionAdapter::getValue( *userData.u, entity, *userData.time ),
                                                         ( *userData.tau ),
                                                         rhs );
         }
   };
};

#include <solvers/pde/tnlLinearSystemAssembler_impl.h>

#endif /* TNLLINEARSYSTEMASSEMBLER_H_ */
