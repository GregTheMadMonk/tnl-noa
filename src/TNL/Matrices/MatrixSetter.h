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
          typename CompressedRowLengthsVector >
class MatrixSetterTraverserUserData
{
   public:
      
      typedef typename CompressedRowLengthsVector::DeviceType DeviceType;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      CompressedRowLengthsVector* rowLengths;

      MatrixSetterTraverserUserData( const DifferentialOperator* differentialOperator,
                                     const BoundaryConditions* boundaryConditions,
                                     CompressedRowLengthsVector* rowLengths )
      : differentialOperator( differentialOperator ),
        boundaryConditions( boundaryConditions ),
        rowLengths( rowLengths )
      {}

};

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowLengthsVector >
class MatrixSetter
{
   public:
   typedef Mesh MeshType;
   typedef Pointers::SharedPointer<  MeshType > MeshPointer;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowLengthsVector::RealType IndexType;
   typedef MatrixSetterTraverserUserData< DifferentialOperator,
                                          BoundaryConditions,
                                          CompressedRowLengthsVector > TraverserUserData;
   typedef Pointers::SharedPointer<  DifferentialOperator, DeviceType > DifferentialOperatorPointer;
   typedef Pointers::SharedPointer<  BoundaryConditions, DeviceType > BoundaryConditionsPointer;
   typedef Pointers::SharedPointer<  CompressedRowLengthsVector, DeviceType > CompressedRowLengthsVectorPointer;

   template< typename EntityType >
   void getCompressedRowLengths( const MeshPointer& meshPointer,
                                  const DifferentialOperatorPointer& differentialOperatorPointer,
                                  const BoundaryConditionsPointer& boundaryConditionsPointer,
                                  CompressedRowLengthsVectorPointer& rowLengthsPointer ) const;

   class TraverserBoundaryEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.rowLengths )[ entity.getIndex() ] =
               userData.boundaryConditions->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
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
            ( *userData.rowLengths )[ entity.getIndex() ] =
               userData.differentialOperator->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }
   };


};

/*
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowLengthsVector >
class MatrixSetter< Meshes::Grid< Dimension, Real, Device, Index >,
                       DifferentialOperator,
                       BoundaryConditions,
                       CompressedRowLengthsVector >
{
   public:
   typedef Meshes::Grid< Dimension, Real, Device, Index > MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename CompressedRowLengthsVector::RealType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef MatrixSetterTraverserUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             CompressedRowLengthsVector > TraverserUserData;

   template< typename EntityType >
   void getCompressedRowLengths( const MeshType& mesh,
                       const DifferentialOperator& differentialOperator,
                       const BoundaryConditions& boundaryConditions,
                       CompressedRowLengthsVector& rowLengths ) const;

   class TraverserBoundaryEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.rowLengths )[ entity.getIndex() ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
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
            ( *userData.rowLengths )[ entity.getIndex() ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }
   };

};
*/

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixSetter_impl.h>
