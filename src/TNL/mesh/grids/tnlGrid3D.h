/***************************************************************************
                          tnlGrid3D.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/tnlGrid.h>
#include <TNL/mesh/grids/tnlGridEntityTopology.h>
#include <TNL/mesh/grids/tnlGridEntityGetter.h>
#include <TNL/mesh/grids/tnlNeighbourGridEntityGetter.h>

namespace TNL {

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 3, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< 3, Real > VertexType;
   typedef tnlStaticVector< 3, Index > CoordinatesType;
   typedef tnlGrid< 3, Real, tnlHost, Index > HostType;
   typedef tnlGrid< 3, Real, tnlCuda, Index > CudaType;
   typedef tnlGrid< 3, Real, Device, Index > ThisType;
 
   static const int meshDimensions = 3;

   template< int EntityDimensions,
             typename Config = tnlGridEntityCrossStencilStorage< 1 > >
   using MeshEntity = tnlGridEntity< ThisType, EntityDimensions, Config >;

   typedef MeshEntity< meshDimensions, tnlGridEntityCrossStencilStorage< 1 > > Cell;
   typedef MeshEntity< meshDimensions - 1 > Face;
   typedef MeshEntity< 1 > Edge;
   typedef MeshEntity< 0 > Vertex;

   static constexpr int getMeshDimensions() { return meshDimensions; };

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize, const Index zSize );

   void setDimensions( const CoordinatesType& );

   __cuda_callable__
   inline const CoordinatesType& getDimensions() const;

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
 
   template< int xPow, int yPow, int zPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const;

 
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;
 
   template< typename GridFunction >
   typename GridFunction::RealType getAbsMax( const GridFunction& f ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getLpNorm( const GridFunction& f,
                                              const typename GridFunction::RealType& p ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
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
 
   IndexType numberOfCells,
          numberOfNxFaces, numberOfNyFaces, numberOfNzFaces, numberOfNxAndNyFaces, numberOfFaces,
          numberOfDxEdges, numberOfDyEdges, numberOfDzEdges, numberOfDxAndDyEdges, numberOfEdges,
          numberOfVertices;

   VertexType origin, proportions;

   IndexType cellZNeighboursStep;
 
   VertexType spaceSteps;
 
   RealType spaceStepsProducts[ 5 ][ 5 ][ 5 ];

   template< typename, typename, int >
   friend class tnlGridEntityGetter;
 
   template< typename, int, typename >
   friend class tnlNeighbourGridEntityGetter;
};

} // namespace TNL

#include <TNL/mesh/grids/tnlGrid3D_impl.h>
