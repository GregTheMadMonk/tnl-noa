#if !defined(_MESH_ENTITY_COBORDER_H_)
#define _MESH_ENTITY_COBORDER_H_

#include "mesh_tags.h"


namespace implementation
{


template<typename MeshConfigTag,
         typename MeshEntityTag,
         typename DimensionTag,
         typename CoborderStorageTag = typename EntityCoborderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::CoborderStorageTag>
class MeshEntityCoborderLayer;


template<typename MeshConfigTag, typename MeshEntityTag>
class MeshEntityCoborder : public MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, DimTag<MeshConfigTag::dimension> >
{
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<true> > : public MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
	typedef MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous> BaseType;

	typedef Mesh<MeshConfigTag> MeshType;

	typedef EntityCoborderTag<MeshConfigTag, MeshEntityTag, DimensionTag> CoborderTag;
	typedef typename CoborderTag::ContainerType                           ContainerType;
	typedef typename CoborderTag::RangeType                               RangeType;
	typedef typename CoborderTag::ConstRangeType                          ConstRangeType;

protected:
	using BaseType::coborderRange;
	RangeType      coborderRange(DimensionTag, MeshType &mesh)       { return RangeType     (mesh.template entities<DimensionTag::value>(), m_coborderEntities); }
	ConstRangeType coborderRange(DimensionTag, MeshType &mesh) const { return ConstRangeType(mesh.template entities<DimensionTag::value>(), m_coborderEntities); }

	using BaseType::coborderContainer;
	ContainerType       &coborderContainer(DimensionTag)             { return m_coborderEntities; }
	const ContainerType &coborderContainer(DimensionTag) const       { return m_coborderEntities; }

	size_t allocatedMemorySize() const                               { return m_coborderEntities.allocatedMemorySize() + BaseType::allocatedMemorySize(); }

private:
	ContainerType m_coborderEntities;
};

template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<false> > : public MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
};

template<typename MeshConfigTag, typename MeshEntityTag>
class MeshEntityCoborderLayer<MeshConfigTag, MeshEntityTag, DimTag<MeshEntityTag::dimension>, StorageTag<false> >
{
protected:
	// These methods are due to 'using BaseType::...;' in the derived classes.
	void coborderRange()     {}
	void coborderContainer() {}

	size_t allocatedMemorySize() const { return 0; }
};


} // namespace implementation


#endif
