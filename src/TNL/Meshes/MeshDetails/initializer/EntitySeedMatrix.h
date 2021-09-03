#pragma once

//#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/IsDynamicTopology.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device>
class MeshTraits;

template< typename MeshConfig,
          typename EntityTopology,
          bool IsDynamicTopology = Topologies::IsDynamicTopology< EntityTopology >::value >
class EntitySeedMatrix;

template< typename MeshConfig,
          typename EntityTopology >
class EntitySeedMatrix< MeshConfig, EntityTopology, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Devices::Host >;

   public:
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
      using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;
      using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

      static constexpr int cornersCount = SubentityTraitsType::count;

      class EntitySeedMatrixSeed
      {
         using RowView = typename SubentityMatrixType::RowView;

         public:
            EntitySeedMatrixSeed( const RowView& matrixRow )
            : row( matrixRow )
            {}

            static constexpr LocalIndexType getCornersCount()
            {
               return cornersCount;
            }

            void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
            {
               TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
               TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
               TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );
               this->row.setColumnIndex( cornerIndex, pointIndex );
            }

            GlobalIndexType getCornerId( const LocalIndexType& cornerIndex ) const
            {
               return this->row.getColumnIndex( cornerIndex );
            }

         private:
            RowView row;
      };

      class ConstEntitySeedMatrixSeed
      {
         using ConstRowView = typename SubentityMatrixType::ConstRowView;

         public:
            ConstEntitySeedMatrixSeed( const ConstRowView& matrixRow )
            : row( matrixRow )
            {}

            static constexpr LocalIndexType getCornersCount()
            {
               return cornersCount;
            }

            GlobalIndexType getCornerId( const LocalIndexType& cornerIndex ) const
            {
               return this->row.getColumnIndex( cornerIndex );
            }

         private:
            ConstRowView row;
      };

      void setDimensions( const GlobalIndexType& entitiesCount, const GlobalIndexType& pointsCount )
      {
         matrix.setDimensions( entitiesCount, pointsCount );

         NeighborCountsArray capacities( entitiesCount );
         capacities.setValue( cornersCount );
         matrix.setRowCapacities( capacities );
      }

      // This method is only here for compatibility with specialization for dynamic entity topologies
      void setEntityCornersCount( const GlobalIndexType& entityIndex, const LocalIndexType& count )
      {}

      // This method is only here for compatibility with specialization for dynamic entity topologies
      void initializeRows()
      {}

      void reset()
      {
         matrix.reset();
      }

      void setSeedIndex( const GlobalIndexType& entityIndex, const LocalIndexType& localIndex, const GlobalIndexType& globalIndex )
      {
         matrix.getRow( entityIndex ).setElement( localIndex, globalIndex, true );
      }

      GlobalIndexType getEntitiesCount() const
      {
         return matrix.getRows();
      }

      constexpr LocalIndexType getEntityCornersCount( const GlobalIndexType& entityIndex ) const
      {
         return cornersCount;
      }

      GlobalIndexType getSeedIndex( const GlobalIndexType& entityIndex, const LocalIndexType& localIndex ) const
      {
         return matrix.getRow( entityIndex ).getColumnIndex( localIndex );
      }

      SubentityMatrixType& getMatrix()
      {
         return matrix;
      }

      const SubentityMatrixType& getMatrix() const
      {
         return matrix;
      }

      NeighborCountsArray getEntityCornerCounts() const
      {
         NeighborCountsArray counts( getEntitiesCount() );
         counts.setValue( cornersCount );
         return counts;
      }

      bool empty() const
      {
         return getEntitiesCount() == 0;
      }

      EntitySeedMatrixSeed getSeed( const GlobalIndexType& entityIndex )
      {
         return EntitySeedMatrixSeed( matrix.getRow( entityIndex ) );
      }

      ConstEntitySeedMatrixSeed getSeed( const GlobalIndexType& entityIndex ) const
      {
         return ConstEntitySeedMatrixSeed( matrix.getRow( entityIndex ) );
      }

   private:
      SubentityMatrixType matrix;
};

template< typename MeshConfig,
          typename EntityTopology >
class EntitySeedMatrix< MeshConfig, EntityTopology, true >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Devices::Host >;

   public:
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;
      using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

      class EntitySeedMatrixSeed
      {
         using RowView = typename SubentityMatrixType::RowView;

         public:
            EntitySeedMatrixSeed( const RowView& matrixRow, const LocalIndexType& corners )
            : row( matrixRow ),
              cornersCount( corners )
            {}

            LocalIndexType getCornersCount() const
            {
               return cornersCount;
            }

            void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
            {
               TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
               TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
               TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );
               this->row.setColumnIndex( cornerIndex, pointIndex );
            }

            GlobalIndexType getCornerId( const LocalIndexType& cornerIndex ) const
            {
               return this->row.getColumnIndex( cornerIndex );
            }

         private:
            RowView row;
            LocalIndexType cornersCount;
      };

      class ConstEntitySeedMatrixSeed
      {
         using ConstRowView = typename SubentityMatrixType::ConstRowView;

         public:
            ConstEntitySeedMatrixSeed( const ConstRowView& matrixRow, const LocalIndexType& corners )
            : row( matrixRow ),
              cornersCount( corners )
            {}

            LocalIndexType getCornersCount() const
            {
               return cornersCount;
            }

            GlobalIndexType getCornerId( const LocalIndexType& cornerIndex ) const
            {
               return this->row.getColumnIndex( cornerIndex );
            }

         private:
            ConstRowView row;
            LocalIndexType cornersCount;
      };

      void setDimensions( const GlobalIndexType& entitiesCount, const GlobalIndexType& pointsCount )
      {
         counts.setSize( entitiesCount );
         matrix.setDimensions( entitiesCount, pointsCount );
      }

      void setEntityCornersCount( const GlobalIndexType& entityIndex, const LocalIndexType& count )
      {
         counts.setElement( entityIndex, count );
      }

      void initializeRows()
      {
         matrix.setRowCapacities( counts );
      }

      void reset()
      {
         matrix.reset();
         counts.reset();
      }

      void setSeedIndex( const GlobalIndexType& entityIndex, const LocalIndexType& localIndex, const GlobalIndexType& globalIndex )
      {
         matrix.getRow( entityIndex ).setElement( localIndex, globalIndex, true );
      }

      GlobalIndexType getEntitiesCount() const
      {
         return matrix.getRows();
      }

      LocalIndexType getEntityCornersCount( const GlobalIndexType& entityIndex ) const
      {
         return counts.getElement( entityIndex );
      }

      GlobalIndexType getSeedIndex( const GlobalIndexType& entityIndex, const LocalIndexType& localIndex ) const
      {
         return matrix.getRow( entityIndex ).getColumnIndex( localIndex );
      }

      SubentityMatrixType& getMatrix()
      {
         return matrix;
      }

      const SubentityMatrixType& getMatrix() const
      {
         return matrix;
      }

      NeighborCountsArray& getEntityCornerCounts()
      {
         return counts;
      }

      const NeighborCountsArray& getEntityCornerCounts() const
      {
         return counts;
      }

      bool empty() const
      {
         return getEntitiesCount() == 0;
      }

      EntitySeedMatrixSeed getSeed( const GlobalIndexType& entityIndex )
      {
         return EntitySeedMatrixSeed( matrix.getRow( entityIndex ), counts[ entityIndex ] );
      }

      ConstEntitySeedMatrixSeed getSeed( const GlobalIndexType& entityIndex ) const
      {
         return ConstEntitySeedMatrixSeed( matrix.getRow( entityIndex ), counts[ entityIndex ] );
      }

   private:
      SubentityMatrixType matrix;
      NeighborCountsArray counts;
};

} // namespace Meshes
} // namespace TNL