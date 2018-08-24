/***************************************************************************
                          SimpleCell.h  -  description
                             -------------------
    begin                : Aug 24, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

template< typename Grid >
class SimpleCell
{
   public:
 
      typedef Grid GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
 
      constexpr static int getMeshDimension() { return GridType::getMeshDimension(); };
 
      constexpr static int getEntityDimension() { return getMeshDimension(); };
 
      typedef SimpleCell< GridType > ThisType;

      __cuda_callable__ inline
      SimpleCell( const GridType& grid ) { this->grid = grid; };
 
      /*__cuda_callable__ inline
      SimpleCell( const GridType& grid,
                  const CoordinatesType& coordinates,
                  const EntityOrientationType& orientation = EntityOrientationType( ( Index ) 0 ),
                  const EntityBasisType& basis = EntityBasisType( ( Index ) 1 ) );*/
 
      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const { return this->coordinates; };
 
      __cuda_callable__ inline
      CoordinatesType& getCoordinates() { return this->coordinates; };
 
      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates ) { this->coordinates = coordinates; };

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      void refresh() { this->entityIndex = this->grid.getEntityIndex( *this ); };

      __cuda_callable__ inline
      IndexType getIndex() const { return this->entityIndex; };
 
      /*__cuda_callable__ inline
      const EntityOrientationType getOrientation() const;
 
      __cuda_callable__ inline
      void setOrientation( const EntityOrientationType& orientation ){};
 
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
 
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};
 
      template< int NeighborEntityDimension = Dimension >
      __cuda_callable__ inline
      const NeighborEntities< NeighborEntityDimension >&
      getNeighborEntities() const;*/
 
      /*__cuda_callable__ inline
      bool isBoundaryEntity() const;
 
      __cuda_callable__ inline
      PointType getCenter() const;
 
      __cuda_callable__ inline
      const RealType& getMeasure() const;
 
      __cuda_callable__ inline
      const PointType& getEntityProportions() const;*/
 
      __cuda_callable__ inline
      const GridType& getMesh() const { return this->grid; };

   protected:
 
      const GridType& grid;
 
      IndexType entityIndex;
 
      CoordinatesType coordinates;
};
