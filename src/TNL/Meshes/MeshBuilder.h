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

namespace TNL {
namespace Meshes {

template< typename Mesh >
class MeshBuilder
{
public:
   using MeshType        = Mesh;
   using MeshTraitsType  = typename MeshType::MeshTraitsType;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
   using PointType       = typename MeshTraitsType::PointType;
   using CellTopology    = typename MeshTraitsType::CellTopology;
   using CellSeedType    = typename MeshTraitsType::CellSeedType;

   void setPointsCount( const GlobalIndexType& points )
   {
      this->points.setSize( points );
      this->pointsSet.setSize( points );
      pointsSet.setValue( false );
   }

   void setCellsCount( const GlobalIndexType& cellsCount )
   {
      this->cellSeeds.setSize( cellsCount );
   }

   GlobalIndexType getPointsCount() const
   {
      return this->points.getSize();
   }

   GlobalIndexType getCellsCount() const
   {
      return this->cellSeeds.getSize();
   }

   void setPoint( GlobalIndexType index,
                  const PointType& point )
   {
      this->points[ index ] = point;
      this->pointsSet[ index ] = true;
   }

   CellSeedType& getCellSeed( GlobalIndexType index )
   {
      return this->cellSeeds[ index ];
   }

   bool build( MeshType& mesh )
   {
      if( ! this->validate() )
         return false;
      mesh.init( this->points, this->cellSeeds );
      return true;
   }

private:
   using PointArrayType    = typename MeshTraitsType::PointArrayType;
   using CellSeedArrayType = typename MeshTraitsType::CellSeedArrayType;
   using BoolVector        = Containers::Vector< bool, Devices::Host, GlobalIndexType >;

   bool validate() const
   {
      if( min( pointsSet ) != true ) {
         std::cerr << "Mesh builder error: Not all points were set." << std::endl;
         return false;
      }

      BoolVector assignedPoints;
      assignedPoints.setLike( pointsSet );
      assignedPoints.setValue( false );

      for( GlobalIndexType i = 0; i < getCellsCount(); i++ ) {
         const auto& cornerIds = this->cellSeeds[ i ].getCornerIds();
         for( LocalIndexType j = 0; j < cornerIds.getSize(); j++ ) {
            assignedPoints[ cornerIds[ j ] ] = true;
            if( cornerIds[ j ] < 0 || getPointsCount() <= cornerIds[ j ] ) {
               std::cerr << "Cell seed " << i << " is referencing unavailable point " << cornerIds[ j ] << std::endl;
               return false;
            }
         }
      }

      if( min( assignedPoints ) != true ) {
         std::cerr << "Mesh builder error: Some points were not used for cells." << std::endl;
         return false;
      }

      return true;
   }

   PointArrayType points;
   CellSeedArrayType cellSeeds;
   BoolVector pointsSet;
};

} // namespace Meshes
} // namespace TNL
