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

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename Mesh >
class MeshBuilder
{
	//static constexpr const char *CLASS_NAME = "MeshBuilder";

   public:
      typedef Mesh                                         MeshType;
      typedef typename MeshType::MeshTraitsType            MeshTraitsType;
      typedef typename MeshTraitsType::GlobalIndexType     GlobalIndexType;
      typedef typename MeshTraitsType::LocalIndexType      LocalIndexType;
      typedef typename MeshTraitsType::PointType           PointType;
      typedef typename MeshTraitsType::CellTopology        CellTopology;
      typedef typename MeshTraitsType::CellSeedType        CellSeedType;

   bool setPointsCount( const GlobalIndexType& points )
   {
      TNL_ASSERT( 0 <= points, std::cerr << "pointsCount = " << points );
      this->points.setSize( points );
      this->pointsSet.setSize( points );
      pointsSet.setValue( false );
      return true;
   }
 
   bool setCellsCount( const GlobalIndexType& cellsCount )
   {
      TNL_ASSERT( 0 <= cellsCount, std::cerr << "cellsCount = " << cellsCount );
      this->cellSeeds.setSize( cellsCount );
      return true;
   }
 
   GlobalIndexType getPointsCount() const { return this->points.getSize(); }
	
   GlobalIndexType getCellsCount() const  { return this->cellSeeds.getSize(); }

   void setPoint( GlobalIndexType index,
                 const PointType& point )
   {
	   Assert( 0 <= index && index < getPointsCount(), std::cerr << "Index = " << index );

      this->points[ index ] = point;
      this->pointsSet[ index ] = true;
   }

   CellSeedType& getCellSeed( GlobalIndexType index )
   {
      TNL_ASSERT( 0 <= index && index < getCellsCount(), std::cerr << "Index = " << index );
 
      return this->cellSeeds[ index ];
   }

   bool build( MeshType& mesh ) const
   {
      if( ! this->validate() )
         return false;
      if( ! mesh.init( this->points, this->cellSeeds ) )
         return false;
      return true;
   }

   private:
      typedef typename MeshTraitsType::PointArrayType    PointArrayType;
      typedef typename MeshTraitsType::CellSeedArrayType CellSeedArrayType;

      bool validate() const
      {
         if( ! allPointsSet() )
         {
            std::cerr << "Mesh builder error: Not all points were set." << std::endl;
            return false;
         }

         std::unordered_set< GlobalIndexType > assignedPoints;
         for( GlobalIndexType i = 0; i < getCellsCount(); i++ )
         {
            auto cornerIds = this->cellSeeds[ i ].getCornerIds();
            for( LocalIndexType j = 0; j < cornerIds.getSize(); j++ )
            {
               assignedPoints.insert( cornerIds[ j ] );
               if( cornerIds[ j ] < 0 || getPointsCount() <= cornerIds[ j ] )
               {
                  std::cerr << "Cell seed " << i << " is referencing unavailable point " << cornerIds[ j ] << std::endl;
                  return false;
               }
            }
         }

         if( (GlobalIndexType) assignedPoints.size() != this->getPointsCount() )
         {
            std::cerr << "Mesh builder error: Some points were not used for cells." << std::endl;
            return false;
         }

         return true;
      }


      bool allPointsSet() const
      {
         for( GlobalIndexType i = 0; i < this->points.getSize(); i++ )
            if( ! this->pointsSet[ i ] )
               return false;
         return true;
      }

      PointArrayType points;
      CellSeedArrayType cellSeeds;
      Containers::Array< bool, Devices::Host, GlobalIndexType > pointsSet;
};

} // namespace Meshes
} // namespace TNL

