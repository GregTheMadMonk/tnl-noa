/***************************************************************************
                          MeshBuilder.h  -  description
                             -------------------
    begin                : Aug 18, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>

namespace TNL {
namespace Meshes {

template< typename Mesh >
class MeshBuilder
{
public:
   using MeshType           = Mesh;
   using MeshTraitsType     = typename MeshType::MeshTraitsType;
   using GlobalIndexType    = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType     = typename MeshTraitsType::LocalIndexType;
   using PointType          = typename MeshTraitsType::PointType;
   using CellTopology       = typename MeshTraitsType::CellTopology;
   using CellSeedMatrixType = typename MeshTraitsType::CellSeedMatrixType;
   using CellSeedType       = typename CellSeedMatrixType::EntitySeedMatrixSeed;
   using FaceSeedMatrixType = typename MeshTraitsType::FaceSeedMatrixType;
   using FaceSeedType       = typename FaceSeedMatrixType::EntitySeedMatrixSeed;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   
   void setEntitiesCount( const GlobalIndexType& points, 
                          const GlobalIndexType& cells = 0,
                          const GlobalIndexType& faces = 0 )
   {
      this->points.setSize( points );
      this->pointsSet.setSize( points );
      pointsSet.setValue( false );

      if( std::is_same< CellTopology, Topologies::Polyhedron >::value )
      {
         this->faceSeeds.setDimensions( faces, points );
         this->cellSeeds.setDimensions( cells, faces );
      }
      else // Topologies other than polyhedrons don't use face seeds
      {
         this->cellSeeds.setDimensions( cells, points );
      }
   }

   void setFaceCornersCounts( const NeighborCountsArray& counts )
   {
      this->faceSeeds.setEntityCornersCounts( counts );
   }

   void setFaceCornersCounts( NeighborCountsArray&& counts )
   {
      this->faceSeeds.setEntityCornersCounts( std::move( counts ) );
   }

   void setCellCornersCounts( const NeighborCountsArray& counts )
   {
      this->cellSeeds.setEntityCornersCounts( counts );
   }

   void setCellCornersCounts( NeighborCountsArray&& counts )
   {
      this->cellSeeds.setEntityCornersCounts( std::move( counts ) );
   }

   GlobalIndexType getPointsCount() const
   {
      return this->points.getSize();
   }

   GlobalIndexType getFacesCount() const
   {
      return this->faceSeeds.getEntitiesCount();
   }

   GlobalIndexType getCellsCount() const
   {
      return this->cellSeeds.getEntitiesCount();
   }

   void setPoint( GlobalIndexType index,
                  const PointType& point )
   {
      this->points[ index ] = point;
      this->pointsSet[ index ] = true;
   }

   FaceSeedType getFaceSeed( GlobalIndexType index )
   {
      return this->faceSeeds.getSeed( index );
   }

   CellSeedType getCellSeed( GlobalIndexType index )
   {
      return this->cellSeeds.getSeed( index );
   }

   bool build( MeshType& mesh )
   {
      if( ! this->validate() )
         return false;
      mesh.init( this->points, this->faceSeeds, this->cellSeeds );
      return true;
   }

private:
   using PointArrayType     = typename MeshTraitsType::PointArrayType;
   using BoolVector         = Containers::Vector< bool, Devices::Host, GlobalIndexType >;

   bool validate() const
   {
      if( min( pointsSet ) != true ) {
         std::cerr << "Mesh builder error: Not all points were set." << std::endl;
         return false;
      }

      BoolVector assignedPoints;
      assignedPoints.setLike( pointsSet );
      assignedPoints.setValue( false );

      if( faceSeeds.empty() )
      {
         for( GlobalIndexType i = 0; i < getCellsCount(); i++ ) {
            const auto cellSeed = this->cellSeeds.getSeed( i );
            for( LocalIndexType j = 0; j < cellSeed.getCornersCount(); j++ ) {
               const GlobalIndexType cornerId = cellSeed.getCornerId( j );
               assignedPoints[ cornerId ] = true;
               if( cornerId < 0 || getPointsCount() <= cornerId ) {
                  std::cerr << "Cell seed " << i << " is referencing unavailable point " << cornerId << std::endl;
                  return false;
               }
            }
         }

         if( min( assignedPoints ) != true ) {
            std::cerr << "Mesh builder error: Some points were not used for cells." << std::endl;
            return false;
         }
      }
      else
      {
         for( GlobalIndexType i = 0; i < getFacesCount(); i++ ) {
            const auto faceSeed = this->faceSeeds.getSeed( i );
            for( LocalIndexType j = 0; j < faceSeed.getCornersCount(); j++ ) {
               const GlobalIndexType cornerId = faceSeed.getCornerId( j );
               if( cornerId < 0 || getPointsCount() <= cornerId ) {
                  std::cerr << "face seed " << i << " is referencing unavailable point " << cornerId << std::endl;
                  return false;
               }
               assignedPoints[ cornerId ] = true;
            }
         }

         if( min( assignedPoints ) != true ) {
            std::cerr << "Mesh builder error: Some points were not used for faces." << std::endl;
            return false;
         }

         BoolVector assignedFaces;
         assignedFaces.setSize( faceSeeds.getEntitiesCount() );
         assignedFaces.setValue( false );

         for( GlobalIndexType i = 0; i < getCellsCount(); i++ ) {
            const auto cellSeed = this->cellSeeds.getSeed( i );
            for( LocalIndexType j = 0; j < cellSeed.getCornersCount(); j++ ) {
               const GlobalIndexType cornerId = cellSeed.getCornerId( j );
               if( cornerId < 0 || getFacesCount() <= cornerId ) {
                  std::cerr << "cell seed " << i << " is referencing unavailable face " << cornerId << std::endl;
                  return false;
               }
               assignedFaces[ cornerId ] = true;
            }
         }

         if( min( assignedFaces ) != true ) {
            std::cerr << "Mesh builder error: Some faces were not used for cells." << std::endl;
            return false;
         }
      }

      return true;
   }

   PointArrayType points;
   FaceSeedMatrixType faceSeeds;
   CellSeedMatrixType cellSeeds;
   BoolVector pointsSet;
};

} // namespace Meshes
} // namespace TNL
