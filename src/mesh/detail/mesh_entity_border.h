#if !defined(_MESH_ENTITY_BORDER_H_)
#define _MESH_ENTITY_BORDER_H_

#include "mesh_tags.h"


namespace implementation
{


template<typename MeshConfigTag,
         typename MeshEntityTag,
         typename DimensionTag,
         typename BorderStorageTag = typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::BorderStorageTag>
class MeshEntityBorderLayer;


template<typename MeshConfigTag, typename MeshEntityTag>
class MeshEntityBorder : public MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, DimTag<MeshEntityTag::dimension - 1> >
{
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<true> > : public MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
	typedef MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous> BaseType;

	typedef Mesh<MeshConfigTag> MeshType;

	typedef EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag> BorderTag;
	typedef typename BorderTag::ContainerType                           ContainerType;
	typedef typename BorderTag::RangeType                               RangeType;
	typedef typename BorderTag::ConstRangeType                          ConstRangeType;

protected:
	using BaseType::borderRange;
	RangeType      borderRange(DimensionTag, MeshType &mesh)       { return RangeType     (mesh.template entities<DimensionTag::value>(), m_borderEntities); }
	ConstRangeType borderRange(DimensionTag, MeshType &mesh) const { return ConstRangeType(mesh.template entities<DimensionTag::value>(), m_borderEntities); }

	using BaseType::borderContainer;
	ContainerType       &borderContainer(DimensionTag)             { return m_borderEntities; }
	const ContainerType &borderContainer(DimensionTag) const       { return m_borderEntities; }

	size_t allocatedMemorySize() const                             { return m_borderEntities.allocatedMemorySize() + BaseType::allocatedMemorySize(); }

private:
	ContainerType m_borderEntities;
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<false> > : public MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
};


template<typename MeshConfigTag, typename MeshEntityTag>
class MeshEntityBorderLayer<MeshConfigTag, MeshEntityTag, DimTag<0>, StorageTag<true> >
{
	typedef DimTag<0> DimensionTag;

	typedef Mesh<MeshConfigTag> MeshType;

	typedef EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag> BorderTag;
	typedef typename BorderTag::ContainerType                           ContainerType;
	typedef typename BorderTag::RangeType                               RangeType;
	typedef typename BorderTag::ConstRangeType                          ConstRangeType;

protected:
	RangeType      borderRange(DimensionTag, MeshType &mesh)       { return RangeType     (mesh.template entities<DimensionTag::value>(), m_borderVertices); }
	ConstRangeType borderRange(DimensionTag, MeshType &mesh) const { return ConstRangeType(mesh.template entities<DimensionTag::value>(), m_borderVertices); }

	ContainerType       &borderContainer(DimensionTag)             { return m_borderVertices; }
	const ContainerType &borderContainer(DimensionTag) const       { return m_borderVertices; }

	size_t allocatedMemorySize() const                             { return m_borderVertices.allocatedMemorySize(); }

private:
	ContainerType m_borderVertices;
};


} // namespace implementation


#endif
