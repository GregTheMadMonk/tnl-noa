#if !defined(_MESH_ENTITY_H_)
#define _MESH_ENTITY_H_

#include <mesh/topology/vertex.h>
#include <mesh/detail/mesh_entity_border.h>
#include <mesh/detail/mesh_entity_coborder.h>
#include <mesh/detail/mesh_pointer.h>
#include <mesh/detail/id.h>
#include <mesh/detail/point.h>


namespace implementation
{


template<typename, typename> class EntityInitializer;

template<typename> class IOReader;
template<typename> class IOWriter;


template<typename MeshConfigTag, typename MeshEntityTag>
class MeshEntity : public MeshPointerProvider<MeshConfigTag>,
                   public MeshEntityBorder<MeshConfigTag, MeshEntityTag>,
                   public MeshEntityCoborder<MeshConfigTag, MeshEntityTag>,
                   public IDProvider<typename MeshConfigTag::IDType, typename MeshConfigTag::GlobalIndexType>
{
	friend class MeshEntityKey<MeshConfigTag, MeshEntityTag>;
	friend class EntityInitializer<MeshConfigTag, MeshEntityTag>;
	friend class IOReader<MeshConfigTag>;
	friend class IOWriter<MeshConfigTag>;

	typedef MeshEntityBorder<MeshConfigTag, MeshEntityTag>   BorderBaseType;
	typedef MeshEntityCoborder<MeshConfigTag, MeshEntityTag> CoborderBaseType;

	typedef EntityBorderTag<MeshConfigTag, MeshEntityTag, DimTag<0> > BorderTag;
	typedef typename BorderTag::ContainerType::DataType  GlobalIndexType;
	typedef typename BorderTag::ContainerType::IndexType LocalIndexType;

	enum { borderVerticesCount = BorderTag::count };

	template<DimensionType dimension>
	struct BorderRangesTag
	{
		typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimTag<dimension> >::RangeType      RangeType;
		typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimTag<dimension> >::ConstRangeType ConstRangeType;
	};

	template<DimensionType dimension>
	struct CoborderRangesTag
	{
		typedef typename EntityCoborderTag<MeshConfigTag, MeshEntityTag, DimTag<dimension> >::RangeType      RangeType;
		typedef typename EntityCoborderTag<MeshConfigTag, MeshEntityTag, DimTag<dimension> >::ConstRangeType ConstRangeType;
	};

public:
	typedef MeshConfigTag MeshConfig;
	typedef MeshEntityTag Tag;

	template<DimensionType dimension> typename BorderRangesTag<dimension>::RangeType      borderEntities()           { return borderRange(DimTag<dimension>(), this->getMesh()); }
	template<DimensionType dimension> typename BorderRangesTag<dimension>::ConstRangeType borderEntities() const     { return borderRange(DimTag<dimension>(), this->getMesh()); }

	template<DimensionType dimension> typename CoborderRangesTag<dimension>::RangeType      coborderEntities()       { return coborderRange(DimTag<dimension>(), this->getMesh()); }
	template<DimensionType dimension> typename CoborderRangesTag<dimension>::ConstRangeType coborderEntities() const { return coborderRange(DimTag<dimension>(), this->getMesh()); }

	size_t allocatedMemorySize() const { return BorderBaseType::allocatedMemorySize() + CoborderBaseType::allocatedMemorySize(); }

protected:
	void setVertex(LocalIndexType localIndex, GlobalIndexType globalIndex)
	{
		assert(0 <= localIndex && localIndex < borderVerticesCount);

		this->borderContainer(DimTag<0>())[localIndex] = globalIndex;
	}
};


template<typename MeshConfigTag>
class MeshEntity<MeshConfigTag, topology::Vertex> : public MeshPointerProvider<MeshConfigTag>,
                                                    public MeshEntityCoborder<MeshConfigTag, topology::Vertex>,
                                                    public IDProvider<typename MeshConfigTag::IDType, typename MeshConfigTag::GlobalIndexType>
{
	friend class EntityInitializer<MeshConfigTag, topology::Vertex>;
	friend class IOReader<MeshConfigTag>;

	typedef MeshEntityCoborder<MeshConfigTag, topology::Vertex> CoborderBaseType;

	template<DimensionType dimension>
	struct CoborderRangesTag
	{
		typedef typename EntityCoborderTag<MeshConfigTag, topology::Vertex, DimTag<dimension> >::RangeType      RangeType;
		typedef typename EntityCoborderTag<MeshConfigTag, topology::Vertex, DimTag<dimension> >::ConstRangeType ConstRangeType;
	};

public:
	typedef MeshConfigTag    MeshConfig;
	typedef topology::Vertex Tag;

	typedef typename MeshTag<MeshConfigTag>::PointType PointType;

	PointType       &getPoint()       { return m_point; }
	const PointType &getPoint() const { return m_point; }

	template<DimensionType dimension> typename CoborderRangesTag<dimension>::RangeType      coborderEntities()       { return coborderRange(DimTag<dimension>(), this->getMesh()); }
	template<DimensionType dimension> typename CoborderRangesTag<dimension>::ConstRangeType coborderEntities() const { return coborderRange(DimTag<dimension>(), this->getMesh()); }

	size_t allocatedMemorySize() const { return CoborderBaseType::allocatedMemorySize(); }

protected:
	void setPoint(const PointType &point) { m_point = point; }

private:
	PointType m_point;
};


} // namespace implementation


#endif
