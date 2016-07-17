/***************************************************************************
                          tnlMeshBuilder.h  -  description
                             -------------------
    begin                : Aug 18, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/traits/tnlMeshTraits.h>

namespace TNL {

template< typename Mesh >
class tnlMeshBuilder
{
	//static constexpr const char *CLASS_NAME = "MeshBuilder";

   public:
      typedef Mesh                                     MeshType;
      typedef typename MeshType::MeshTraits            MeshTraits;
      typedef typename MeshTraits::GlobalIndexType     GlobalIndexType;
      typedef typename MeshTraits::LocalIndexType      LocalIndexType;
      typedef typename MeshTraits::PointType           PointType;
      typedef typename MeshTraits::CellTopology        CellTopology;
      typedef typename MeshTraits::CellSeedType        CellSeedType;

   bool setPointsCount( const GlobalIndexType& points )
   {
      tnlAssert( 0 <= points, cerr << "pointsCount = " << points );
      this->points.setSize( points );
      this->pointsSet.setSize( points );
      pointsSet.setValue( false );
      return true;
   }
 
   bool setCellsCount( const GlobalIndexType& cellsCount )
   {
      tnlAssert( 0 <= cellsCount, cerr << "cellsCount = " << cellsCount );
      this->cellSeeds.setSize( cellsCount );
      return true;
   }
 
   GlobalIndexType getPointsCount() const { return this->points.getSize(); }
	
   GlobalIndexType getCellsCount() const  { return this->cellSeeds.getSize(); }

   void setPoint( GlobalIndexType index,
                 const PointType& point )
   {
	tnlAssert( 0 <= index && index < getPointsCount(), cerr << "Index = " << index );

        this->points[ index ] = point;
        this->pointsSet[ index ] = true;
   }

   CellSeedType& getCellSeed( GlobalIndexType index )
   {
      tnlAssert( 0 <= index && index < getCellsCount(), cerr << "Index = " << index );
 
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
      typedef typename MeshTraits::PointArrayType    PointArrayType;
      typedef typename MeshTraits::CellSeedArrayType CellSeedArrayType;

      bool validate() const
      {
         if( !allPointsSet() )
         {
            cerr << "Mesh builder error: Not all points were set." << endl;
            return false;
         }

         for( GlobalIndexType i = 0; i < getCellsCount(); i++ )
         {
            auto cornerIds = this->cellSeeds[ i ].getCornerIds();
            for( LocalIndexType j = 0; j < cornerIds.getSize(); j++ )
               if( cornerIds[ j ] < 0 || getPointsCount() <= cornerIds[ j ] )
               {
                  cerr << "Cell seed " << i << " is referencing unavailable point " << cornerIds[ j ] << endl;
                  return false;
               }
         }
         return true;
      }


      bool allPointsSet() const
      {
         for( GlobalIndexType i = 0; i < this->points.getSize(); i++ )
            if (! this->pointsSet[ i ] )
               return false;
            return true;
      }

      PointArrayType points;
      CellSeedArrayType cellSeeds;
      tnlArray< bool, tnlHost, GlobalIndexType > pointsSet;
};

} // namespace TNL

