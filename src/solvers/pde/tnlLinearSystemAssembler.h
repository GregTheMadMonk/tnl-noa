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
                                                 const Real& timeDiscretisationCoefficient,
                                                 const DifferentialOperator& differentialOperator,
                                                 const BoundaryConditions& boundaryConditions,
                                                 const RightHandSide& rightHandSide,
                                                 DofVector& u,
                                                 Matrix& matrix,
                                                 DofVector& b )
      : time( &time ),
        tau( &tau ),
        timeDiscretisationCoefficient( &timeDiscretisationCoefficient ),
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
#ifdef HAVE_CUDA
         __host__ __device__
#endif
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const IndexType index )
         {
            typename MatrixType::MatrixRow matrixRow = userData.matrix->getRow( index );
            userData.boundaryConditions->updateLinearSystem( *userData.time + *userData.tau,
                                                             mesh,
                                                             index,
                                                             *userData.u,
                                                             *userData.b,
                                                             matrixRow );
         }

   };

   class TraverserInteriorEntitiesProcessor
   {
      public:

         template< int EntityDimensions >
#ifdef HAVE_CUDA
         __host__ __device__
#endif
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

            typename MatrixType::MatrixRow matrixRow = userData.matrix->getRow( index );
            userData.differentialOperator->updateLinearSystem( *userData.time,
                                                               *userData.tau,
                                                               mesh,
                                                               index,
                                                               *userData.u,
                                                               *userData.b,
                                                               matrixRow );
            //userData.matrix->addElement( index, index, 1.0, 1.0 );
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

   tnlLinearSystemAssembler()
   : timeDiscretisationCoefficient( 1.0 ){}

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

   /****
    * TODO: Fix this. Somehow.
    */
   void setTimeDiscretisationCoefficient( const Real& c )
   {
      this->timeDiscretisationCoefficient = c;
   }

   class TraverserBoundaryEntitiesProcessor
   {
      public:

#ifdef HAVE_CUDA
         __host__ __device__
#endif
         static void processCell( const MeshType& mesh,
                                  TraverserUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            //printf( "index = %d \n", index );
             ( *userData.b )[ index ] = 0.0;
            typename MatrixType::MatrixRow matrixRow = userData.matrix->getRow( index );
            userData.boundaryConditions->updateLinearSystem( *userData.time + *userData.tau,
                                                             mesh,
                                                             index,
                                                             coordinates,
                                                             *userData.u,
                                                             *userData.b,
                                                             matrixRow );
         }

#ifdef HAVE_CUDA
         __host__ __device__
#endif
         static void processFace( const MeshType& mesh,
                                  TraverserUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            //printf( "index = %d \n", index );
            // printf("Matrix assembler: Index = %d \n", index );
            ( *userData.b )[ index ] = 0.0;
            typename MatrixType::MatrixRow matrixRow = userData.matrix->getRow( index );
            userData.boundaryConditions->updateLinearSystem( *userData.time,
                                                             mesh,
                                                             index,
                                                             coordinates,
                                                             *userData.u,
                                                             *userData.b,
                                                             matrixRow );
            //printf( "BC: index = %d, b = %f \n", index, ( *userData.b )[ index ] );
         }


   };

   class TraverserInteriorEntitiesProcessor
   {
      public:

      /****
       *
       * TODO: FIX THIS. The assembler is not designed properly for the stationary problems!!!
       *
       */
#ifdef HAVE_CUDA
         __host__ __device__
#endif
         static void processCell( const MeshType& mesh,
                                  TraverserUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            //printf( "index = %d \n", index );
            typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
            ( *userData.b )[ index ] = 0.0; /*( *userData.timeDiscretisationCoefficient) * ( *userData.u )[ index ] +
                                  ( *userData.tau ) * FunctionAdapter::getValue( mesh,
                                                             *userData.rightHandSide,
                                                             index,
                                                             coordinates,
                                                             *userData.time );*/
            
            typename MatrixType::MatrixRow matrixRow = userData.matrix->getRow( index );
            userData.differentialOperator->updateLinearSystem( *userData.time,
                                                               *userData.tau,
                                                               mesh,
                                                               index,
                                                               coordinates,
                                                               *userData.u,
                                                               *userData.b,
                                                               matrixRow );
            /*if( *userData.timeDiscretisationCoefficient != 0.0 )
               userData.matrix->addElementFast( index,
                                                index,
                                                *userData.timeDiscretisationCoefficient,
                                                1.0 );*/
            
            const RealType& rhs = FunctionAdapter::getValue( mesh,
                                                             *userData.rightHandSide,
                                                             index,
                                                             coordinates,
                                                             *userData.time );
            TimeDiscretisation::applyTimeDiscretisation( *userData.matrix,
                                                         ( *userData.b )[ index ],
                                                         index,
                                                         ( *userData.u )[ index ],
                                                         ( *userData.tau ),
                                                         rhs );
            //printf( "IC: index = %d, b = %f \n", index, ( *userData.b )[ index ] );
         }

#ifdef HAVE_CUDA
         __host__ __device__
#endif
         static void processFace( const MeshType& mesh,
                                  TraverserUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            //printf( "index = %d \n", index );
            // printf("Matrix assembler: Index = %d \n", index );
            typedef tnlFunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
            ( *userData.b )[ index ] = 0.0; /*( *userData.timeDiscretisationCoefficient) * ( *userData.u )[ index ] +
                                  ( *userData.tau ) * FunctionAdapter::getValue( mesh,
                                                             *userData.rightHandSide,
                                                             index,
                                                             coordinates,
                                                             *userData.time );*/

            typename MatrixType::MatrixRow matrixRow = userData.matrix->getRow( index );
            userData.differentialOperator->updateLinearSystem( *userData.time,
                                                               *userData.tau,
                                                               mesh,
                                                               index,
                                                               coordinates,
                                                               *userData.u,
                                                               *userData.b,
                                                               matrixRow );
            /*if( *userData.timeDiscretisationCoefficient != 0.0 )
               userData.matrix->addElementFast( index,
                                                index,
                                                *userData.timeDiscretisationCoefficient,
                                                1.0 );*/
            
            const RealType& rhs = FunctionAdapter::getValue( mesh,
                                                             *userData.rightHandSide,
                                                             index,
                                                             coordinates,
                                                             *userData.time );
            TimeDiscretisation::applyTimeDiscretisation( *userData.matrix,
                                                         ( *userData.b )[ index ],
                                                         index,
                                                         ( *userData.u )[ index ],
                                                         ( *userData.tau ),
                                                         rhs );

         }
   };

   protected:

   Real timeDiscretisationCoefficient;
};

#include <solvers/pde/tnlLinearSystemAssembler_impl.h>

#endif /* TNLLINEARSYSTEMASSEMBLER_H_ */
