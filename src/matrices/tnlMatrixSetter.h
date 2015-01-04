/***************************************************************************
                          tnlMatrixSetter.h  -  description
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

#ifndef TNLMATRIXSETTER_H_
#define TNLMATRIXSETTER_H_

template< typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowLengthsVector >
class tnlMatrixSetterTraversalUserData
{
   public:

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      RowLengthsVector* rowLengths;

      tnlMatrixSetterTraversalUserData( const DifferentialOperator& differentialOperator,
                                        const BoundaryConditions& boundaryConditions,
                                        RowLengthsVector& rowLengths )
      : differentialOperator( &differentialOperator ),
        boundaryConditions( &boundaryConditions ),
        rowLengths( &rowLengths )
      {};

};

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowLengthsVector >
class tnlMatrixSetter
{
   public:
   typedef Mesh MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename RowLengthsVector::RealType IndexType;
   typedef tnlMatrixSetterTraversalUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             RowLengthsVector > TraversalUserData;

   template< int EntityDimensions >
   void getRowLengths( const MeshType& mesh,
                       DifferentialOperator& differentialOperator,
                       BoundaryConditions& boundaryConditions,
                       RowLengthsVector& rowLengths ) const;


   class TraversalBoundaryEntitiesProcessor
   {
      public:

         template< int EntityDimension >
#ifdef HAVE_CUDA
         __device__ __host__
#endif
         static void processEntity( const MeshType& mesh,
                                    TraversalUserData& userData,
                                    const IndexType index )
         {
            ( *userData.rowLengths )[ index ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, index );
         }

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:

         template< int EntityDimensions >
#ifdef HAVE_CUDA
         __device__ __host__
#endif
         static void processEntity( const MeshType& mesh,
                                    TraversalUserData& userData,
                                    const IndexType index )
         {
            ( *userData.rowLengths )[ index ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, index );
         }

   };

};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowLengthsVector >
class tnlMatrixSetter< tnlGrid< Dimensions, Real, Device, Index >,
                       DifferentialOperator,
                       BoundaryConditions,
                       RowLengthsVector >
{
   public:
   typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename RowLengthsVector::RealType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef tnlMatrixSetterTraversalUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             RowLengthsVector > TraversalUserData;

   template< int EntityDimensions >
   void getRowLengths( const MeshType& mesh,
                       const DifferentialOperator& differentialOperator,
                       const BoundaryConditions& boundaryConditions,
                       RowLengthsVector& rowLengths ) const;

   class TraversalBoundaryEntitiesProcessor
   {
      public:

#ifdef HAVE_CUDA
         __device__ __host__
#endif
         static void processCell( const MeshType& mesh,
                                  TraversalUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            ( *userData.rowLengths )[ index ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, index, coordinates );
         }

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:

#ifdef HAVE_CUDA
         __device__ __host__
#endif
         static void processCell( const MeshType& mesh,
                                  TraversalUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            ( *userData.rowLengths )[ index ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, index, coordinates );
         }

   };

};

#include <implementation/matrices/tnlMatrixSetter_impl.h>

#endif /* TNLMATRIXSETTER_H_ */
