#if !defined(_MESH_LAYER_H_)
#define _MESH_LAYER_H_

#include <mesh/common/common.h>
#include <mesh/detail/mesh_tags.h>
#include <mesh/detail/mesh_entity.h>


namespace implementation
{


template<typename MeshConfigTag,
         typename DimensionTag,
         typename EntityStorageTag = typename EntityTag<MeshConfigTag, DimensionTag>::EntityStorageTag>
class MeshLayer;


template<typename MeshConfigTag>
class MeshLayers : public MeshLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag>
{
};


template<typename MeshConfigTag, typename DimensionTag>
class MeshLayer<MeshConfigTag, DimensionTag, StorageTag<true> > : public MeshLayer<MeshConfigTag, typename DimensionTag::Previous>
{
	typedef MeshLayer<MeshConfigTag, typename DimensionTag::Previous> BaseType;

	typedef EntityTag<MeshConfigTag, DimensionTag> Tag;
	typedef typename Tag::ContainerType            ContainerType;
	typedef typename Tag::RangeType                RangeType;
	typedef typename Tag::ConstRangeType           ConstRangeType;

protected:
	using BaseType::entityRange;
	RangeType      entityRange(DimensionTag)                 { return RangeType(m_entities); }
	ConstRangeType entityRange(DimensionTag) const           { return ConstRangeType(m_entities); }

	using BaseType::entityContainer;
	ContainerType       &entityContainer(DimensionTag)       { return m_entities; }
	const ContainerType &entityContainer(DimensionTag) const { return m_entities; }

	size_t allocatedMemorySize() const
	{
		size_t memorySize = m_entities.allocatedMemorySize();
		for (typename ContainerType::IndexType i = 0; i < m_entities.size(); i++)
			memorySize += m_entities[i].allocatedMemorySize();

		return memorySize + BaseType::allocatedMemorySize();
	}

private:
	ContainerType m_entities;
};


template<typename MeshConfigTag, typename DimensionTag>
class MeshLayer<MeshConfigTag, DimensionTag, StorageTag<false> > : public MeshLayer<MeshConfigTag, typename DimensionTag::Previous>
{
};


template<typename MeshConfigTag>
class MeshLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag, StorageTag<true> > : public MeshLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag::Previous>
{
	typedef typename MeshTag<MeshConfigTag>::MeshDimTag               DimensionTag;
	typedef MeshLayer<MeshConfigTag, typename DimensionTag::Previous> BaseType;

	typedef EntityTag<MeshConfigTag, DimensionTag> Tag;
	typedef typename Tag::ContainerType            ContainerType;
	typedef typename Tag::RangeType                RangeType;
	typedef typename Tag::ConstRangeType           ConstRangeType;

protected:
	using BaseType::entityRange;
	RangeType      entityRange(DimensionTag)                 { return RangeType(m_cells); }
	ConstRangeType entityRange(DimensionTag) const           { return ConstRangeType(m_cells); }

	using BaseType::entityContainer;
	ContainerType       &entityContainer(DimensionTag)       { return m_cells; }
	const ContainerType &entityContainer(DimensionTag) const { return m_cells; }

	size_t allocatedMemorySize() const
	{
		size_t memorySize = m_cells.allocatedMemorySize();
		for (typename ContainerType::IndexType i = 0; i < m_cells.size(); i++)
			memorySize += m_cells[i].allocatedMemorySize();

		return memorySize + BaseType::allocatedMemorySize();
	}

private:
	ContainerType m_cells;
};


template<typename MeshConfigTag>
class MeshLayer<MeshConfigTag, DimTag<0>, StorageTag<true> >
{
	typedef DimTag<0> DimensionTag;

	typedef EntityTag<MeshConfigTag, DimensionTag> Tag;
	typedef typename Tag::ContainerType            ContainerType;
	typedef typename Tag::RangeType                RangeType;
	typedef typename Tag::ConstRangeType           ConstRangeType;

protected:
	RangeType      entityRange(DimensionTag)                 { return RangeType(m_vertices); }
	ConstRangeType entityRange(DimensionTag) const           { return ConstRangeType(m_vertices); }

	ContainerType       &entityContainer(DimensionTag)       { return m_vertices; }
	const ContainerType &entityContainer(DimensionTag) const { return m_vertices; }

	size_t allocatedMemorySize() const
	{
		size_t memorySize = m_vertices.allocatedMemorySize();
		for (typename ContainerType::IndexType i = 0; i < m_vertices.size(); i++)
			memorySize += m_vertices[i].allocatedMemorySize();

		return memorySize;
	}

private:
	ContainerType m_vertices;
};


} // namespace implementation


#endif
