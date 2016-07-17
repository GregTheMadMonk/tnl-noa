/***************************************************************************
                          tnlMatrixSetter.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

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

   template< typename EntityType >
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

         template< typename EntityType >
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

} // namespace TNL

#include <matrices/tnlMatrixSetter_impl.h>
