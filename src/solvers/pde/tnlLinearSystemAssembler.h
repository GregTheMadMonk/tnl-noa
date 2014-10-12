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

template< typename Real,
          typename DofVector,
          typename Matrix,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class tnlLinearSystemAssemblerTraversalUserData
{
   public:

      const Real &time;

      const Real &tau;

      const DifferentialOperator& differentialOperator;

      const BoundaryConditions& boundaryConditions;

      const RightHandSide& rightHandSide;

      DofVector &u, &b;

      Matrix &matrix;

      tnlLinearSystemAssemblerTraversalUserData( const Real& time,
                                                 const Real& tau,
                                                 const DifferentialOperator& differentialOperator,
                                                 const BoundaryConditions& boundaryConditions,
                                                 const RightHandSide& rightHandSide,
                                                 DofVector& u,
                                                 DofVector& b,
                                                 Matrix& matrix )
      : time( time ),
        tau( tau ),
        differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rightHandSide( rightHandSide ),
        u( u ),
        b( b ),
        matrix( matrix )
      {};

   protected:

};


template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
class tnlLinearSystemAssembler
{
   public:
   typedef Mesh MeshType;
   typedef typename DofVector::RealType RealType;
   typedef typename DofVector::DeviceType DeviceType;
   typedef typename DofVector::IndexType IndexType;
   typedef Matrix MatrixType;
   typedef tnlLinearSystemAssemblerTraversalUserData< RealType,
                                                      DofVector,
                                                      DifferentialOperator,
                                                      BoundaryConditions,
                                                      RightHandSide > TraversalUserData;

   template< int EntityDimensions >
   void assembly( const RealType& time,
                  const MeshType& mesh,
                  const DifferentialOperator& differentialOperator,
                  const BoundaryConditions& boundaryConditions,
                  const RightHandSide& rightHandSide,
                  DofVector& u,
                  MatrixType& matrix,
                  DofVector& b ) const;

   class TraversalBoundaryEntitiesProcessor
   {
      public:

         template< int EntityDimension >
#ifdef HAVE_CUDA
         __host__ __device__
#endif
         void processEntity( const MeshType& mesh,
                             TraversalUserData& userData,
                             const IndexType index )
         {
            userData.boundaryConditions.updateLinearSystem( userData.time,
                                                            mesh,
                                                            index,
                                                            userData.u,
                                                            userData.matrix,
                                                            userData.b );
         }

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:

         template< int EntityDimensions >
#ifdef HAVE_CUDA
         __host__ __device__
#endif
         void processEntity( const MeshType& mesh,
                             TraversalUserData& userData,
                             const IndexType index )
         {
            userData.differentialOperator.updateLinearSystem( mesh,
                                                              index,
                                                              userData.u,
                                                              userData.matrix,
                                                              userData.b,
                                                              userData.time );
            userData.b[ index ] += userData.tau * userData.rightHandSide.getValue( mesh.getEntityCenter< EntityDimensions >( index ),
                                                                                   userData.time );
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
          typename Matrix >
class tnlLinearSystemAssembler< tnlGrid< Dimensions, Real, Device, Index >,
                                DofVector,
                                DifferentialOperator,
                                BoundaryConditions,
                                RightHandSide,
                                Matrix >
{
   public:
   typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
   typedef typename DofVector::RealType RealType;
   typedef typename DofVector::DeviceType DeviceType;
   typedef typename DofVector::IndexType IndexType;
   typedef Matrix MatrixType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef tnlLinearSystemAssemblerTraversalUserData< RealType,
                                                      DofVector,
                                                      DifferentialOperator,
                                                      BoundaryConditions,
                                                      RightHandSide > TraversalUserData;

   template< int EntityDimensions >
   void assembly( const RealType& time,
                  const MeshType& mesh,
                  const DifferentialOperator& differentialOperator,
                  const BoundaryConditions& boundaryConditions,
                  const RightHandSide& rightHandSide,
                  DofVector& u,
                  MatrixType& matrix,
                  DofVector& b ) const;

   class TraversalBoundaryEntitiesProcessor
   {
      public:

         template< int EntityDimension >
#ifdef HAVE_CUDA
         __host__ __device__
#endif
         void processEntity( const MeshType& mesh,
                             TraversalUserData& userData,
                             const IndexType index,
                             const CoordinatesType& coordinates )
         {
            userData.boundaryConditions.updateLinearSystem( userData.time,
                                                            mesh,
                                                            index,
                                                            coordinates,
                                                            userData.u,
                                                            userData.matrix,
                                                            userData.b );
         }

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:

         template< int EntityDimensions >
#ifdef HAVE_CUDA
         __host__ __device__
#endif
         void processEntity( const MeshType& mesh,
                             TraversalUserData& userData,
                             const IndexType index,
                             const CoordinatesType& coordinates )
         {
            userData.differentialOperator.updateLinearSystem( mesh,
                                                              index,
                                                              coordinates,
                                                              userData.u,
                                                              userData.matrix,
                                                              userData.b,
                                                              userData.time );
            userData.b[ index ] += userData.tau *
                                   userData.rightHandSide.getValue( mesh.getEntityCenter< EntityDimensions >( index ),
                                                                    userData.time );
         }

   };
};

#include <implementation/solvers/pde/tnlLinearSystemAssembler_impl.h>

#endif /* TNLLINEARSYSTEMASSEMBLER_H_ */
