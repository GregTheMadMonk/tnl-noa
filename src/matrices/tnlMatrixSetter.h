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

#pragma once

template< typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
class tnlMatrixSetterTraversalUserData
{
   public:
      
      typedef typename CompressedRowsLengthsVector::DeviceType DeviceType;
      typedef tnlSharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef tnlSharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef tnlSharedPointer< CompressedRowsLengthsVector, DeviceType > CompressedRowsLengthsVectorPointer;


      const DifferentialOperatorPointer differentialOperatorPointer;

      const BoundaryConditionsPointer boundaryConditionsPointer;

      CompressedRowsLengthsVectorPointer rowLengthsPointer;

      tnlMatrixSetterTraversalUserData( const DifferentialOperatorPointer& differentialOperatorPointer,
                                        const BoundaryConditionsPointer& boundaryConditionsPointer,
                                        CompressedRowsLengthsVectorPointer& rowLengthsPointer )
      : differentialOperatorPointer( differentialOperatorPointer ),
        boundaryConditionsPointer( boundaryConditionsPointer ),
        rowLengthsPointer( rowLengthsPointer )
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
   typedef tnlSharedPointer< MeshType > MeshPointer;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowsLengthsVector::RealType IndexType;
   typedef tnlMatrixSetterTraversalUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             CompressedRowsLengthsVector > TraversalUserData;
   typedef tnlSharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
   typedef tnlSharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
   typedef tnlSharedPointer< CompressedRowsLengthsVector, DeviceType > CompressedRowsLengthsVectorPointer;

   template< typename EntityType >
   void getCompressedRowsLengths( const MeshPointer& meshPointer,
                       DifferentialOperatorPointer& differentialOperatorPointer,
                       BoundaryConditionsPointer& boundaryConditionsPointer,
                       CompressedRowsLengthsVectorPointer& rowLengthsPointer ) const;

   class TraversalBoundaryEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraversalUserData& userData,                                    
                                    const EntityType& entity )
         {
            userData.rowLengthsPointer.template modifyData< DeviceType >()[ entity.getIndex() ] =
                     userData.boundaryConditionsPointer.template getData< DeviceType >().getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:
         
         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraversalUserData& userData,
                                    const EntityType& entity )
         {
            userData.rowLengthsPointer.template modifyData< DeviceType >()[ entity.getIndex() ] =
               userData.differentialOperatorPointer.template getData< DeviceType >().getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }
   };


};

/*
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

   template< typename EntityType >
   void getCompressedRowsLengths( const MeshType& mesh,
                       const DifferentialOperator& differentialOperator,
                       const BoundaryConditions& boundaryConditions,
                       CompressedRowsLengthsVector& rowLengths ) const;

   class TraversalBoundaryEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraversalUserData& userData,                                    
                                    const EntityType& entity )
         {
            ( *userData.rowLengths )[ entity.getIndex() ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }

   };

   class TraversalInteriorEntitiesProcessor
   {
      public:
         
         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraversalUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.rowLengths )[ entity.getIndex() ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }
   };

};
*/

#include <matrices/tnlMatrixSetter_impl.h>
