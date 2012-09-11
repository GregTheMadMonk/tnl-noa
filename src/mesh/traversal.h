#if !defined(_MESH_TRAVERSAL_H_)
#define _MESH_TRAVERSAL_H_

#include "global/static_assert.h"
#include "detail/mesh_tags.h"
#include "mesh.h"
#include "information.h"


template<typename MeshType, DimensionType dimension>
class Entity
{
public:
	typedef typename implementation::EntityTag<typename MeshType::Config, implementation::DimTag<dimension> >::Type Type;
};


template<typename MeshType, DimensionType dimension>
class EntityRange
{
public:
	typedef typename implementation::EntityTag<typename MeshType::Config, implementation::DimTag<dimension> >::ConstRangeType Type;
};


template<typename EntityType, DimensionType dimension>
class BorderRange
{
public:
	typedef typename implementation::EntityBorderTag<typename EntityType::MeshConfig, typename EntityType::Tag, implementation::DimTag<dimension> >::ConstRangeType Type;
};


template<typename EntityType, DimensionType dimension>
class CoborderRange
{
public:
	typedef typename implementation::EntityCoborderTag<typename EntityType::MeshConfig, typename EntityType::Tag, implementation::DimTag<dimension> >::ConstRangeType Type;
};


template<DimensionType dimension, typename MeshType>
typename EntityRange<MeshType, dimension>::Type entities(const MeshType &mesh)
{
	STATIC_ASSERT((EntitiesAvailable<MeshType, dimension>::value), "Mesh configuration does not allow access to requested mesh entities");

	return mesh.template entities<dimension>();
}


template<DimensionType dimension, typename EntityType>
typename BorderRange<EntityType, dimension>::Type borderEntities(const EntityType &entity)
{
	STATIC_ASSERT((BorderEntitiesAvailable<EntityType, dimension>::value), "Mesh configuration does not allow access to requested border entities");

	return entity.template borderEntities<dimension>();
}


template<DimensionType dimension, typename EntityType>
typename CoborderRange<EntityType, dimension>::Type coborderEntities(const EntityType &entity)
{
	STATIC_ASSERT((CoborderEntitiesAvailable<EntityType, dimension>::value), "Mesh configuration does not allow access to requested coborder entities");

	return entity.template coborderEntities<dimension>();
}


#endif
