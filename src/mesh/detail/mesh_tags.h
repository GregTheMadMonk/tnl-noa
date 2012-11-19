#if !defined(_MESH_TAGS_H_)
#define _MESH_TAGS_H_

#include <mesh/common/common.h>
#include <mesh/detail/container.h>
#include <mesh/detail/range.h>


namespace implementation
{


template<typename, typename>      class MeshEntity;
template<typename, typename>      class MeshEntityKey;
template<typename, DimensionType> class Point;


template<typename MeshConfigTag>
class MeshTag
{
public:
	enum { dimension = MeshConfigTag::Cell::dimension };

	typedef DimTag<dimension> MeshDimTag;

	typedef Point<typename MeshConfigTag::NumericType, MeshConfigTag::dimWorld> PointType;
};


template<typename MeshConfigTag, typename DimensionTag>
class MeshEntitiesTag
{
public:
	typedef typename topology::BorderEntities<typename MeshConfigTag::Cell, DimensionTag::value>::Tag Tag;
};

template<typename MeshConfigTag>
class MeshEntitiesTag<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag>
{
public:
	typedef typename MeshConfigTag::Cell Tag;
};


template<typename MeshConfigTag, typename DimensionTag>
class EntityTag
{
	enum { storageEnabled = config::EntityStorage<MeshConfigTag, DimensionTag::value>::enabled };

	typedef typename MeshConfigTag::GlobalIndexType                    IndexType;
	typedef typename MeshEntitiesTag<MeshConfigTag, DimensionTag>::Tag MeshEntityTag;
	typedef MeshEntityKey<MeshConfigTag, MeshEntityTag>                Key;

public:
	typedef MeshEntityTag                                              Tag;
	typedef MeshEntity<MeshConfigTag, Tag>                             Type;

	typedef StorageTag<storageEnabled>                                 EntityStorageTag;

	typedef Container<Type, IndexType>                                 ContainerType;
	typedef Range<ContainerType>                                       RangeType;
	typedef ConstRange<ContainerType>                                  ConstRangeType;
	typedef IndexedUniqueContainer<Type, IndexType, Key>               UniqueContainerType;
};


// Border entities of a mesh entity
template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityBorderTag
{
	enum { storageEnabled = config::BorderStorage<MeshConfigTag, MeshEntityTag, DimensionTag::value>::enabled };

	typedef typename MeshConfigTag::GlobalIndexType                         GlobalIndexType;
	typedef typename MeshConfigTag::LocalIndexType                          LocalIndexType;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::RangeType      EntityRangeType;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::ConstRangeType EntityConstRangeType;
	typedef topology::BorderEntities<MeshEntityTag, DimensionTag::value>    BorderTag;

public:
	typedef MeshEntity<MeshConfigTag, MeshEntityTag>                        EntityType;
	typedef typename BorderTag::Tag                                         BorderEntityTag;
	typedef MeshEntity<MeshConfigTag, BorderEntityTag>                      BorderEntityType;

	typedef StorageTag<storageEnabled>                                      BorderStorageTag;

	enum { count = BorderTag::count };

	typedef StaticContainer<GlobalIndexType, LocalIndexType, count>         ContainerType;
	typedef IndirectRange<EntityRangeType, ContainerType>                   RangeType;
	typedef ConstIndirectRange<EntityConstRangeType, ContainerType>         ConstRangeType;

	template<LocalIndexType borderEntityIndex, LocalIndexType borderEntityVertexIndex>
	class Vertex
	{
	public:
		enum { index = topology::BorderEntityVertex<MeshEntityTag, BorderEntityTag, borderEntityIndex, borderEntityVertexIndex>::index };
	};
};


// Coborder entities of a mesh entity
template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityCoborderTag
{
	enum { storageEnabled = config::CoborderStorage<MeshConfigTag, MeshEntityTag, DimensionTag::value>::enabled };

	typedef typename MeshConfigTag::GlobalIndexType                         GlobalIndexType;
	typedef typename MeshConfigTag::LocalIndexType                          LocalIndexType;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::RangeType      EntityRangeType;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::ConstRangeType EntityConstRangeType;

public:
	typedef MeshEntity<MeshConfigTag, MeshEntityTag>                        EntityType;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::Tag            CoborderEntityTag;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::Type           CoborderEntityType;

	typedef StorageTag<storageEnabled>                                      CoborderStorageTag;

	typedef Container<GlobalIndexType, LocalIndexType>                      ContainerType;
	typedef IndirectRange<EntityRangeType, ContainerType>                   RangeType;
	typedef ConstIndirectRange<EntityConstRangeType, ContainerType>         ConstRangeType;
	typedef GrowableContainer<GlobalIndexType>                              GrowableContainerType;
};


} // namespace implementation


#endif
