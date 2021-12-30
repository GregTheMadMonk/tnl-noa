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

#include <numeric>   // std::iota
#include <vector>

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
   using PointArrayType     = typename MeshTraitsType::PointArrayType;
   using BoolVector         = Containers::Vector< bool, Devices::Host, GlobalIndexType >;
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

   void deduplicatePoints( const double numericalThreshold = 1e-9 )
   {
      // prepare vector with an identity permutationutation
      std::vector< GlobalIndexType > permutation( points.getSize() );
      std::iota( permutation.begin(), permutation.end(), (GlobalIndexType) 0 );

      // workaround for lexicographical sorting
      // FIXME: https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/-/issues/79
      auto lexless = [numericalThreshold, this] (const GlobalIndexType& a, const GlobalIndexType& b) -> bool
      {
         const PointType& left = this->points[ a ];
         const PointType& right = this->points[ b ];
         for( LocalIndexType i = 0; i < PointType::getSize(); i++ )
            if( TNL::abs( left[ i ] - right[ i ] ) > numericalThreshold )
               return left[ i ] < right[ i ];
         return false;
      };

      // sort points in lexicographical order
      std::stable_sort( permutation.begin(), permutation.end(), lexless );

      // old -> new index mapping for points
      std::vector< GlobalIndexType > points_perm_to_new( points.getSize() );

      // find duplicate points
      GlobalIndexType uniquePointsCount = 0;
      // first index is unique
      points_perm_to_new[ permutation[ 0 ] ] = uniquePointsCount++;
      for( GlobalIndexType i = 1; i < points.getSize(); i++ ) {
         const PointType& curr = points[ permutation[ i ] ];
         const PointType& prev = points[ permutation[ i - 1 ] ];
         if( maxNorm( curr - prev ) > numericalThreshold )
            // unique point
            points_perm_to_new[ permutation[ i ] ] = uniquePointsCount++;
         else
            // duplicate point - use previous index
            points_perm_to_new[ permutation[ i ] ] = uniquePointsCount - 1;
      }

      // if all points are unique, we are done
      if( uniquePointsCount == points.getSize() )
         return;

      std::cout << "Found " << points.getSize() - uniquePointsCount << " duplicate points (total " << points.getSize() << ", unique " << uniquePointsCount << ")" << std::endl;

      // copy this->points and this->pointsSet, drop duplicate points
      // (trying to do this in-place is not worth it, since even Array::reallocate
      // needs to allocate a temporary array and copy the elements)
      PointArrayType newPoints( uniquePointsCount );
      BoolVector newPointsSet( uniquePointsCount );
      std::vector< GlobalIndexType > points_old_to_new( points.getSize() );
      // TODO: this can almost be parallelized, except we have multiple writes for the duplicate points
      for( std::size_t i = 0; i < points_perm_to_new.size(); i++ ) {
         const GlobalIndexType oldIndex = permutation[ i ];
         const GlobalIndexType newIndex = points_perm_to_new[ oldIndex ];
         newPoints[ newIndex ] = points[ oldIndex ];
         newPointsSet[ newIndex ] = pointsSet[ oldIndex ];
         points_old_to_new[ oldIndex ] = newIndex;
      }
      points = std::move( newPoints );
      pointsSet = std::move( newPointsSet );
      // reset permutation and points_perm_to_new - we need just points_old_to_new further on
      permutation.clear();
      permutation.shrink_to_fit();
      points_perm_to_new.clear();
      points_perm_to_new.shrink_to_fit();

      auto remap_matrix = [uniquePointsCount, &points_old_to_new] ( auto& seeds )
      {
         // TODO: parallelize (we have the IndexPermutationApplier)
         for( GlobalIndexType i = 0; i < seeds.getEntitiesCount(); i++ ) {
            auto seed = seeds.getSeed( i );
            for( LocalIndexType j = 0; j < seed.getCornersCount(); j++ ) {
               const GlobalIndexType newIndex = points_old_to_new[ seed.getCornerId( j ) ];
               seed.setCornerId( j, newIndex );
            }
         }
         // update the number of columns of the matrix
         seeds.getMatrix().setColumnsWithoutReset( uniquePointsCount );
      };

      // remap points in this->faceSeeds or this->cellSeeds
      if( faceSeeds.empty() )
         remap_matrix( cellSeeds );
      else
         remap_matrix( faceSeeds );
   }

   bool build( MeshType& mesh )
   {
      if( ! this->validate() )
         return false;
      pointsSet.reset();
      mesh.init( this->points, this->faceSeeds, this->cellSeeds );
      return true;
   }

private:
   bool validate() const
   {
      // verify that matrix dimensions are consistent with points
      if( faceSeeds.empty() ) {
         // no face seeds - cell seeds refer to points
         if( cellSeeds.getMatrix().getColumns() != points.getSize() ) {
            std::cerr << "Mesh builder error: Inconsistent size of the cellSeeds matrix (it has "
                      << cellSeeds.getMatrix().getColumns() << " columns, but there are " << points.getSize()
                      << " points)." << std::endl;
            return false;
         }
      }
      else {
         // cell seeds refer to faces and face seeds refer to points
         if( cellSeeds.getMatrix().getColumns() != faceSeeds.getMatrix().getRows() ) {
            std::cerr << "Mesh builder error: Inconsistent size of the cellSeeds matrix (it has "
                      << cellSeeds.getMatrix().getColumns() << " columns, but there are " << faceSeeds.getMatrix().getRows()
                      << " faces)." << std::endl;
            return false;
         }
         if( faceSeeds.getMatrix().getColumns() != points.getSize() ) {
            std::cerr << "Mesh builder error: Inconsistent size of the faceSeeds matrix (it has "
                      << faceSeeds.getMatrix().getColumns() << " columns, but there are " << points.getSize()
                      << " points)." << std::endl;
            return false;
         }
      }

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
