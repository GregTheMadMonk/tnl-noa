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

#include <functors/tnlFunctorAdapter.h>

template< typename Real,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
class tnlLinearSystemAssemblerTraverserUserData
{
   public:
      typedef Matrix MatrixType;
      typedef typename Matrix::DeviceType DeviceType;

      const Real* time;

      const Real* tau;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      DofVector *u, *b;

      Matrix *matrix;

      const Real* timeDiscretisationCoefficient;

      tnlLinearSystemAssemblerTraverserUserData( const Real& time,
                                                 const Real& tau,
                                                 const DifferentialOperator& differentialOperator,
                                                 const BoundaryConditions& boundaryConditions,
                                                 const RightHandSide& rightHandSide,
                                                 DofVector& u,
                                                 Matrix& matrix,
                                                 DofVector& b )
      : time( &time ),
        tau( &tau ),
        differentialOperator( &differentialOperator ),
        boundaryConditions( &boundaryConditions ),
        rightHandSide( &rightHandSide ),
        u( &u ),
        b( &b ),
        matrix( &matrix )
      {};

   protected:

};


template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename Matrix >
class tnlLinearSystemAssembler
{
   public:
   typedef Mesh MeshType;
   typedef typename DofVector::RealType RealType;
   typedef typename DofVector::DeviceType DeviceType;
   typedef typename DofVector::IndexType IndexType;
   typedef Matrix MatrixType;
   typedef tnlLinearSystemAssemblerTraverserUserData< RealType,
                                                      DofVector,
                                                      DifferentialOperator,
                                                      BoundaryConditions,
                                                      RightHandSide,
                                                      MatrixType > TraverserUserData;

   template< int EntityDimensions >
   void assembly( const RealType& time,
                  const RealType& tau,
                  const MeshType& mesh,
                  const DifferentialOperator& differentialOperator,
                  const BoundaryConditions& boundaryConditions,
                  const RightHandSide& rightHandSide,
                  DofVector& u,
                  MatrixType& matrix,
                  DofVector& b ) const;

   class TraverserBoundaryEntitiesProcessor
   {
      public:

         template< int EntityDimension >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const IndexType index )
         {            
            userData.boundaryConditions->updateLinearSystem( *userData.time + *userData.tau,
                                                             mesh,
                                                             index,
                                                             *userData.u,
                                                             *userData.b,
                                                             *userData.matrix );
         }

   };

   class TraverserInteriorEntitiesProcessor
   {
      public:

         template< int EntityDimensions >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const IndexType index )
         {
            typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
            ( *userData.b )[ index ] = 0.0;/*( *userData.u )[ index ] +
                     ( *userData.tau ) * FunctionAdapter::getValue( mesh,
                                                                    *userData.rightHandSide,
                                                                    index,
                                                                    *userData.time );*/

            userData.differentialOperator->updateLinearSystem( *userData.time,
                                                               *userData.tau,
                                                               mesh,
                                                               index,
                                                               *userData.u,
                                                               *userData.b,
                                                               *userData.matrix );

            const RealType& rhs = FunctionAdapter::getValue( mesh,
                                                             *userData.rightHandSide,
                                                             index,
                                                             *userData.time );
            TimeDiscretisation::applyTimeDiscretisation( *userData.matrix,
                                                         ( *userData.b )[ index ],
                                                         index,
                                                         ( *userData.u )[ index ],
                                                         ( *userData.tau ),
                                                         rhs );
            
         }
   };
};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename Matrix >
class tnlLinearSystemAssembler< tnlGrid< Dimensions, Real, Device, Index >,
                                DofVector,
                                DifferentialOperator,
                                BoundaryConditions,
                                RightHandSide,
                                TimeDiscretisation,
                                Matrix >
{
   public:
   typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
   typedef typename DofVector::RealType RealType;
   typedef typename DofVector::DeviceType DeviceType;
   typedef typename DofVector::IndexType IndexType;
   typedef Matrix MatrixType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef tnlLinearSystemAssemblerTraverserUserData< RealType,
                                                      DofVector,
                                                      DifferentialOperator,
                                                      BoundaryConditions,
                                                      RightHandSide,
                                                      MatrixType > TraverserUserData;

   template< int EntityDimensions >
   void assembly( const RealType& time,
                  const RealType& tau,
                  const MeshType& mesh,
                  const DifferentialOperator& differentialOperator,
                  const BoundaryConditions& boundaryConditions,
                  const RightHandSide& rightHandSide,
                  DofVector& u,
                  MatrixType& matrix,
                  DofVector& b ) const;

   class TraverserBoundaryEntitiesProcessor
   {
      public:
         
         template< typename EntityType >         
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const IndexType index,
                                    const EntityType& entity )
         {
             ( *userData.b )[ index ] = 0.0;           
             userData.boundaryConditions->updateLinearSystem
               ( *userData.time + *userData.tau,
                 mesh,
                 index,
                 entity,
                 *userData.u,
                 *userData.b,
                 *userData.matrix );
         }
   };

   class TraverserInteriorEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const IndexType index,
                                    const EntityType& entity )
         {
            ( *userData.b )[ index ] = 0.0;            
            userData.differentialOperator->updateLinearSystem
               ( *userData.time,
                 *userData.tau,
                 mesh,
                 index,
                 entity,
                 *userData.u,
                 *userData.b,
                 *userData.matrix );
            
            typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
            const RealType& rhs = FunctionAdapter::getValue
               ( mesh,
                 *userData.rightHandSide,
                 index,
                 entity,
                 *userData.time );
            TimeDiscretisation::applyTimeDiscretisation( *userData.matrix,
                                                         ( *userData.b )[ index ],
                                                         index,
                                                         ( *userData.u )[ index ],
                                                         ( *userData.tau ),
                                                         rhs );
         }
   };
};

#include <solvers/pde/tnlLinearSystemAssembler_impl.h>

#endif /* TNLLINEARSYSTEMASSEMBLER_H_ */
