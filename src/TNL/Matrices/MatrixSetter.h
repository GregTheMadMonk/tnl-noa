/***************************************************************************
                          MatrixSetter.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
class MatrixSetterTraversalUserData
{
   public:
      
      typedef typename CompressedRowsLengthsVector::DeviceType DeviceType;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      CompressedRowsLengthsVector* rowLengths;

      MatrixSetterTraversalUserData( const DifferentialOperator* differentialOperator,
                                     const BoundaryConditions* boundaryConditions,
                                     CompressedRowsLengthsVector* rowLengths )
      : differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rowLengths( rowLengths )
      {}

};

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
class MatrixSetter
{
   public:
   typedef Mesh MeshType;
   typedef SharedPointer< MeshType > MeshPointer;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowsLengthsVector::RealType IndexType;
   typedef MatrixSetterTraversalUserData< DifferentialOperator,
                                          BoundaryConditions,
                                          CompressedRowsLengthsVector > TraversalUserData;
   typedef SharedPointer< DifferentialOperator, DeviceType > DifferentialOperatorPointer;
   typedef SharedPointer< BoundaryConditions, DeviceType > BoundaryConditionsPointer;
   typedef SharedPointer< CompressedRowsLengthsVector, DeviceType > CompressedRowsLengthsVectorPointer;

   template< typename EntityType >
   void getCompressedRowsLengths( const MeshPointer& meshPointer,
                                  const DifferentialOperatorPointer& differentialOperatorPointer,
                                  const BoundaryConditionsPointer& boundaryConditionsPointer,
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

/*
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
class MatrixSetter< Meshes::Grid< Dimensions, Real, Device, Index >,
                       DifferentialOperator,
                       BoundaryConditions,
                       CompressedRowsLengthsVector >
{
   public:
   typedef Meshes::Grid< Dimensions, Real, Device, Index > MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowsLengthsVector::RealType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef MatrixSetterTraversalUserData< DifferentialOperator,
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

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixSetter_impl.h>
