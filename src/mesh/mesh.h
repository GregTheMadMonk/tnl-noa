#if !defined(_MESH_H_)
#define _MESH_H_

#include "io/io.h"
#include "detail/mesh_layer.h"
#include "detail/mesh_initializer.h"


namespace implementation
{


template<typename MeshConfigTag>
class Mesh : public MeshLayers<MeshConfigTag>
{
	template<typename, typename, typename> friend class MeshInitializerLayer;
	friend class IOReader<MeshConfigTag>;

	typedef typename EntityTag<MeshConfigTag, DimTag<0> >::RangeType                                  VertexRangeType;
	typedef typename EntityTag<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag>::RangeType CellRangeType;

	template<DimensionType dim>
	struct EntityRangesTag
	{
		typedef typename EntityTag<MeshConfigTag, DimTag<dim> >::RangeType      RangeType;
		typedef typename EntityTag<MeshConfigTag, DimTag<dim> >::ConstRangeType ConstRangeType;
	};

public:
	typedef MeshConfigTag Config;

	enum { dimension = MeshTag<MeshConfigTag>::dimension };

	void load(const char *filename);
	void write(const char *filename) const;

	void load(IOReader<MeshConfigTag> &reader);
	void write(IOWriter<MeshConfigTag> &writer) const;

	template<DimensionType dim> typename EntityRangesTag<dim>::RangeType      entities()       { return this->entityRange(DimTag<dim>()); }
	template<DimensionType dim> typename EntityRangesTag<dim>::ConstRangeType entities() const { return this->entityRange(DimTag<dim>()); }

	size_t memoryRequirement() const { return sizeof(*this) + this->allocatedMemorySize(); }

private:
	void init();
};


template<typename MeshConfigTag>
void Mesh<MeshConfigTag>::load(const char *filename)
{
	typename IOFactory<MeshConfigTag>::ReaderAutoPtr reader = IOFactory<MeshConfigTag>::getReader(filename);
	load(*reader);
}

template<typename MeshConfigTag>
void Mesh<MeshConfigTag>::write(const char *filename) const
{
	typename IOFactory<MeshConfigTag>::WriterAutoPtr writer = IOFactory<MeshConfigTag>::getWriter(filename);
	write(*writer);
}

template<typename MeshConfigTag>
void Mesh<MeshConfigTag>::load(IOReader<MeshConfigTag> &reader)
{
	reader.readMesh(*this);

	init();
}

template<typename MeshConfigTag>
void Mesh<MeshConfigTag>::write(IOWriter<MeshConfigTag> &writer) const
{
	writer.writeMesh(*this);
}

template<typename MeshConfigTag>
void Mesh<MeshConfigTag>::init()
{
	MeshInitializer<MeshConfigTag> meshInitializer(*this);
	meshInitializer.initMesh();
}


}


using implementation::Mesh;


#endif
