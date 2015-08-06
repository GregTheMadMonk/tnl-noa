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
          typename CompressedRowsLengthsVector >
class tnlMatrixSetterTraversalUserData
{
   public:

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      CompressedRowsLengthsVector* rowLengths;

      tnlMatrixSetterTraversalUserData( const DifferentialOperator& differentialOperator,
                                        const BoundaryConditions& boundaryConditions,
                                        CompressedRowsLengthsVector& rowLengths )
      : differentialOperator( &differentialOperator ),
        boundaryConditions( &boundaryConditions ),
        rowLengths( &rowLengths )
      {};

};

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
class tnlMatrixSetter
{
   public:
   typedef Mesh MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowsLengthsVector::RealType IndexType;
   typedef tnlMatrixSetterTraversalUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             CompressedRowsLengthsVector > TraversalUserData;

   template< int EntityDimensions >
   void getCompressedRowsLengths( const MeshType& mesh,
                       DifferentialOperator& differentialOperator,
                       BoundaryConditions& boundaryConditions,
                       CompressedRowsLengthsVector& rowLengths ) const;


   class TraversalBoundaryEntitiesProcessor
   {
      public:

         template< int EntityDimension >
         __cuda_callable__
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
         __cuda_callable__
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
          typename CompressedRowsLengthsVector >
class tnlMatrixSetter< tnlGrid< Dimensions, Real, Device, Index >,
                       DifferentialOperator,
                       BoundaryConditions,
                       CompressedRowsLengthsVector >
{
   public:
   typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowsLengthsVector::RealType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef tnlMatrixSetterTraversalUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             CompressedRowsLengthsVector > TraversalUserData;

   template< int EntityDimensions >
   void getCompressedRowsLengths( const MeshType& mesh,
                       const DifferentialOperator& differentialOperator,
                       const BoundaryConditions& boundaryConditions,
                       CompressedRowsLengthsVector& rowLengths ) const;

   class TraversalBoundaryEntitiesProcessor
   {
      public:

         __cuda_callable__
         static void processCell( const MeshType& mesh,
                                  TraversalUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            ( *userData.rowLengths )[ index ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, index, coordinates );
         }

         __cuda_callable__
         static void processFace( const MeshType& mesh,
                                  TraversalUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
             //printf("Matrix setter: Index = %d \n", index );
            ( *userData.rowLengths )[ index ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, index, coordinates );
         }
         

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:

         __cuda_callable__
         static void processCell( const MeshType& mesh,
                                  TraversalUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            ( *userData.rowLengths )[ index ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, index, coordinates );
         }
         
         __cuda_callable__
         static void processFace( const MeshType& mesh,
                                  TraversalUserData& userData,
                                  const IndexType index,
                                  const CoordinatesType& coordinates )
         {
            // printf("Matrix setter: Index = %d \n", index );
            ( *userData.rowLengths )[ index ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, index, coordinates );
         }
         

   };

};

#include <matrices/tnlMatrixSetter_impl.h>

#endif /* TNLMATRIXSETTER_H_ */
