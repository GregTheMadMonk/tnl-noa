/***************************************************************************
                          tnlGrid1D.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef SRC_MESH_TNLGRID1D_H_
#define SRC_MESH_TNLGRID1D_H_

#include <mesh/tnlGrid.h>
#include <core/tnlLogger.h>
#include <mesh/grids/tnlGridEntityTopology.h>
#include <mesh/grids/tnlGridEntityGetter.h>
#include <mesh/grids/tnlNeighbourGridEntityGetter.h>
#include <mesh/grids/tnlGridEntity.h>
#include <mesh/grids/tnlGridEntityConfig.h>

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 1, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< 1, Real > VertexType;
   typedef tnlStaticVector< 1, Index > CoordinatesType;   
   typedef tnlGrid< 1, Real, tnlHost, Index > HostType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > CudaType;
   typedef tnlGrid< 1, Real, Device, Index > ThisType;
   
   static const int meshDimensions = 1;
   
   template< int EntityDimensions, 
             typename Config = tnlGridEntityCrossStencilStorage< 1 > >
   using MeshEntity = tnlGridEntity< ThisType, EntityDimensions, Config >;
     
   typedef MeshEntity< meshDimensions, tnlGridEntityCrossStencilStorage< 1 > > Cell;
   typedef MeshEntity< 0 > Face;
   typedef MeshEntity< 0 > Vertex;

   static constexpr int getDimensionsCount() { return meshDimensions; };
   
   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__ inline
   const CoordinatesType& getDimensions() const;

   void setDomain( const VertexType& origin,
                   const VertexType& proportions );

   __cuda_callable__
   inline const VertexType& getOrigin() const;

   __cuda_callable__
   inline const VertexType& getProportions() const;
   
   template< typename EntityType >
   __cuda_callable__
   inline IndexType getEntitiesCount() const;
   
   template< typename EntityType >
   __cuda_callable__
   inline EntityType getEntity( const IndexType& entityIndex ) const;
   
   template< typename EntityType >
   __cuda_callable__
   inline Index getEntityIndex( const EntityType& entity ) const;
   
   template< typename EntityType >
   __cuda_callable__
   RealType getEntityMeasure( const EntityType& entity ) const;
   
   __cuda_callable__
   RealType getCellMeasure() const;
   
   __cuda_callable__
   inline VertexType getSpaceSteps() const;

   template< int xPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const;
   
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;


   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   /****
    *  Method for saving the object to a file as a binary data
    */
   bool save( tnlFile& file ) const;

   /****
    *  Method for restoring the object from a file
    */
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   bool writeMesh( const tnlString& fileName,
                   const tnlString& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;

   void writeProlog( tnlLogger& logger );

   protected:

   void computeSpaceSteps();

   CoordinatesType dimensions;
   
   IndexType numberOfCells, numberOfVertices;

   VertexType origin, proportions;
   
   VertexType spaceSteps;
   
   RealType spaceStepsProducts[ 5 ];
};

#include <mesh/grids/tnlGrid1D_impl.h>

#endif /* SRC_MESH_TNLGRID1D_H_ */
