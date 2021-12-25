#pragma once 
 
template< typename GridEntity >
class TestNeighborGridEntitiesStorage
{  
   public:
      
      __cuda_callable__
      TestNeighborGridEntitiesStorage( const GridEntity& entity )
      : entity( entity )
      {}
      
      const GridEntity& entity;
};

template< typename Grid,          
          int EntityDimension >
class TestGridEntity
{
};


template< int Dimension,
          typename Real,
          typename Device,
          typename Index,          
          int EntityDimension >
class TestGridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension >
{
   public:
      static const int entityDimension = EntityDimension;
};

/****
 * Specializations for cells
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
class TestGridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension >
{
   public:
      
      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
      //typedef Config ConfigType;
      
      static const int meshDimension = GridType::meshDimension;
      
      static const int entityDimension = meshDimension;

      constexpr static int getDimensions() { return entityDimension; };
      
      constexpr static int getDimension() { return meshDimension; };
      
      
      typedef Containers::StaticVector< meshDimension, IndexType > EntityOrientationType;
      typedef Containers::StaticVector< meshDimension, IndexType > EntityBasisType;
      typedef TestNeighborGridEntitiesStorage< TestGridEntity > NeighborGridEntitiesStorageType;
      
      __cuda_callable__ inline
      TestGridEntity( const GridType& grid )
      : grid( grid ),
        /*entityIndex( -1 ),*/
        neighborEntitiesStorage( *this )
      {
      }
      
      
      __cuda_callable__ inline
      TestGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) )
      : grid( grid ),
        /*entityIndex( -1 ),
        coordinates( coordinates ),*/
        neighborEntitiesStorage( *this )
        {
        }

      

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      NeighborGridEntitiesStorageType neighborEntitiesStorage;
      
};





