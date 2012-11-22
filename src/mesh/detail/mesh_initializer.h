#if !defined(_MESH_INITIALIZER_H_)
#define _MESH_INITIALIZER_H_

#include <mesh/global/static_loop.h>
#include <mesh/detail/mesh_tags.h>
#include <mesh/detail/mesh_entity_key.h>
#include <mesh/detail/mesh_pointer.h>


namespace implementation
{


template<typename MeshConfigTag,
         typename DimensionTag,
         typename EntityStorageTag = typename EntityTag<MeshConfigTag, DimensionTag>::EntityStorageTag>
class MeshInitializerLayer;

template<typename MeshConfigTag,
         typename MeshEntityTag,
         typename DimensionTag,
         typename BorderStorageTag = typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::BorderStorageTag,
         typename CoborderStorageTag = typename EntityCoborderTag<MeshConfigTag, typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::BorderEntityTag, DimTag<MeshEntityTag::dimension> >::CoborderStorageTag>
class EntityInitializerLayer;

template<typename MeshConfigTag,
         typename MeshEntityTag,
         typename DimensionTag,
         typename CoborderStorageTag = typename EntityCoborderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::CoborderStorageTag>
class EntityInitializerCoborderLayer;

template<typename MeshConfigTag,
         typename MeshEntityTag>
class EntityInitializer;


template<typename MeshConfigTag>
class MeshInitializer : public MeshInitializerLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag>
{
	typedef MeshInitializerLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag> BaseType;

	typedef Mesh<MeshConfigTag> MeshType;

public:
	MeshInitializer(MeshType &mesh)
	{
		this->setMesh(mesh);
	}

	void initMesh()
	{
		BaseType::createEntitiesFromCells();
		BaseType::createEntityInitializers();
		BaseType::initEntities(*this);
	}
};


template<typename MeshConfigTag, typename DimensionTag>
class MeshInitializerLayer<MeshConfigTag, DimensionTag, StorageTag<true> > : public MeshInitializerLayer<MeshConfigTag, typename DimensionTag::Previous>
{
	typedef MeshInitializerLayer<MeshConfigTag, typename DimensionTag::Previous> BaseType;

	typedef EntityTag<MeshConfigTag, DimensionTag> Tag;
	typedef typename Tag::Tag                      MeshEntityTag;
	typedef typename Tag::Type                     EntityType;
	typedef typename Tag::RangeType                RangeType;
	typedef typename Tag::UniqueContainerType      UniqueContainerType;
	typedef typename Tag::ContainerType::IndexType GlobalIndexType;

	typedef MeshInitializer<MeshConfigTag>                                 MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, typename MeshConfigTag::Cell> CellInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag>                EntityInitializerType;
	typedef Container<EntityInitializerType, GlobalIndexType>              EntityInitializerContainerType;

	typedef typename CellInitializerType::template BorderEntities<DimensionTag>::ContainerType BorderEntitiesContainerType;

public:
	using BaseType::findEntityIndex;
	GlobalIndexType findEntityIndex(EntityType &entity) const                        { return m_uniqueContainer.find(entity); }

	using BaseType::getEntityInitializer;
	EntityInitializerType &getEntityInitializer(DimensionTag, GlobalIndexType index) { return m_entityInitializerContainer[index]; }

protected:
	void createEntitiesFromCells(const CellInitializerType &cellInitializer)
	{
		BorderEntitiesContainerType borderEntities;
		cellInitializer.template createBorderEntities<DimensionTag>(borderEntities);

		for (typename BorderEntitiesContainerType::IndexType i = 0; i < borderEntities.size(); i++)
			m_uniqueContainer.insert(borderEntities[i]);

		BaseType::createEntitiesFromCells(cellInitializer);
	}

	void createEntityInitializers()
	{
		m_entityInitializerContainer.create(m_uniqueContainer.size());

		BaseType::createEntityInitializers();
	}

	void initEntities(MeshInitializerType &meshInitializer)
	{
		m_uniqueContainer.copy(this->getMesh().entityContainer(DimensionTag()));
		m_uniqueContainer.free();

		RangeType entityRange = this->getMesh().template entities<DimensionTag::value>();
		for (typename RangeType::IndexType i = 0; i < entityRange.size(); i++)
		{
			EntityInitializerType &entityInitializer = m_entityInitializerContainer[i];
			entityInitializer.init(entityRange[i], i);
			entityInitializer.initEntity(meshInitializer);
		}

		m_entityInitializerContainer.free();

		BaseType::initEntities(meshInitializer);
	}

private:
	UniqueContainerType m_uniqueContainer;
	EntityInitializerContainerType m_entityInitializerContainer;
};


template<typename MeshConfigTag, typename DimensionTag>
class MeshInitializerLayer<MeshConfigTag, DimensionTag, StorageTag<false> > : public MeshInitializerLayer<MeshConfigTag, typename DimensionTag::Previous>
{
};


template<typename MeshConfigTag>
class MeshInitializerLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag, StorageTag<true> > : public MeshInitializerLayer<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag::Previous>
{
	typedef typename MeshTag<MeshConfigTag>::MeshDimTag                          DimensionTag;
	typedef MeshInitializerLayer<MeshConfigTag, typename DimensionTag::Previous> BaseType;

	typedef EntityTag<MeshConfigTag, DimensionTag> Tag;
	typedef typename Tag::Tag                      MeshEntityTag;
	typedef typename Tag::RangeType                RangeType;
	typedef typename Tag::ContainerType::IndexType GlobalIndexType;

	typedef MeshInitializer<MeshConfigTag>                  MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> CellInitializerType;
	typedef Container<CellInitializerType, GlobalIndexType> CellInitializerContainerType;

public:
	using BaseType::getEntityInitializer;
	CellInitializerType &getEntityInitializer(DimensionTag, GlobalIndexType index) { return m_cellInitializerContainer[index]; }

protected:
	void createEntitiesFromCells()
	{
		RangeType cellRange = this->getMesh().template entities<DimensionTag::value>();

		m_cellInitializerContainer.create(cellRange.size());
		for (typename RangeType::IndexType i = 0; i < cellRange.size(); i++)
		{
			CellInitializerType &cellInitializer = m_cellInitializerContainer[i];
			cellInitializer.init(cellRange[i], i);

			BaseType::createEntitiesFromCells(cellInitializer);
		}
	}

	void initEntities(MeshInitializerType &meshInitializer)
	{
		for (typename CellInitializerContainerType::IndexType i = 0; i < m_cellInitializerContainer.size(); i++)
			m_cellInitializerContainer[i].initEntity(meshInitializer);

		m_cellInitializerContainer.free();

		BaseType::initEntities(meshInitializer);
	}

private:
	CellInitializerContainerType m_cellInitializerContainer;
};


template<typename MeshConfigTag>
class MeshInitializerLayer<MeshConfigTag, DimTag<0>, StorageTag<true> > : public MeshPointerProvider<MeshConfigTag>
{
	typedef DimTag<0> DimensionTag;

	typedef EntityTag<MeshConfigTag, DimensionTag> Tag;
	typedef typename Tag::Tag                      MeshEntityTag;
	typedef typename Tag::RangeType                RangeType;
	typedef typename Tag::ContainerType::IndexType GlobalIndexType;

	typedef typename EntityTag<MeshConfigTag, typename MeshTag<MeshConfigTag>::MeshDimTag>::Type CellType;

	typedef MeshInitializer<MeshConfigTag>                                 MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, typename MeshConfigTag::Cell> CellInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag>                VertexInitializerType;
	typedef Container<VertexInitializerType, GlobalIndexType>              VertexInitializerContainerType;

public:
	VertexInitializerType &getEntityInitializer(DimensionTag, GlobalIndexType index) { return m_vertexInitializerContainer[index]; }

protected:
	void findEntityIndex() const                              {} // This method is due to 'using BaseType::findEntityIndex;' in the derived class.
	void createEntitiesFromCells(const CellInitializerType &) {}

	void createEntityInitializers()
	{
		m_vertexInitializerContainer.create(this->getMesh().template entities<DimensionTag::value>().size());
	}

	void initEntities(MeshInitializerType &meshInitializer)
	{
		RangeType vertexRange = this->getMesh().template entities<DimensionTag::value>();
		for (typename RangeType::IndexType i = 0; i < vertexRange.size(); i++)
		{
			VertexInitializerType &vertexInitializer = m_vertexInitializerContainer[i];
			vertexInitializer.init(vertexRange[i], i);
			vertexInitializer.initEntity(meshInitializer);
		}

		m_vertexInitializerContainer.free();
	}

private:
	VertexInitializerContainerType m_vertexInitializerContainer;
};


template<typename MeshConfigTag, typename MeshEntityTag>
class EntityInitializer : public EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimTag<MeshEntityTag::dimension - 1> >,
                          public EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, typename MeshTag<MeshConfigTag>::MeshDimTag>
{
	typedef EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimTag<MeshEntityTag::dimension - 1> >               BaseType;
	typedef EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, typename MeshTag<MeshConfigTag>::MeshDimTag> CoborderBaseType;

	typedef typename EntityTag<MeshConfigTag, DimTag<MeshEntityTag::dimension> >::Type                     EntityType;
	typedef typename EntityTag<MeshConfigTag, DimTag<MeshEntityTag::dimension> >::ContainerType::IndexType GlobalIndexType;

	typedef EntityBorderTag<MeshConfigTag, MeshEntityTag, DimTag<0> > BorderVertexTag;
	typedef typename BorderVertexTag::ContainerType::DataType         VertexGlobalIndexType;
	typedef typename BorderVertexTag::ContainerType::IndexType        VertexLocalIndexType;

	typedef MeshInitializer<MeshConfigTag> MeshInitializerType;

	template<typename> class BorderEntitiesCreator;

public:
	template<typename DimensionTag> struct BorderEntities { typedef typename BorderEntitiesCreator<DimensionTag>::BorderContainerType ContainerType; };

	EntityInitializer() : m_entity(0), m_entityIndex(-1) {}

	void init(EntityType &entity, GlobalIndexType entityIndex)
	{
		m_entity = &entity;
		m_entityIndex = entityIndex;
	}

	void initEntity(MeshInitializerType &meshInitializer)
	{
		assert(m_entity);

		m_entity->setMesh(meshInitializer.getMesh());
		m_entity->setID(m_entityIndex);

		initEntityCoborder();
		initEntityBorder(meshInitializer);
	}

	template<typename DimensionTag>
	void createBorderEntities(typename BorderEntities<DimensionTag>::ContainerType &borderEntities) const
	{
		BorderEntitiesCreator<DimensionTag>::createBorderEntities(borderEntities, *m_entity);
	}

	GlobalIndexType getEntityIndex() const
	{
		assert(m_entityIndex >= 0);
		return m_entityIndex;
	}

	template<typename BorderDimensionTag>
	typename EntityBorderTag<MeshConfigTag, MeshEntityTag, BorderDimensionTag>::ContainerType &getEntityBorderContainer(BorderDimensionTag)
	{
		return m_entity->borderContainer(BorderDimensionTag());
	}

	template<typename CoborderDimensionTag>
	typename EntityCoborderTag<MeshConfigTag, MeshEntityTag, CoborderDimensionTag>::ContainerType &getEntityCoborderContainer(CoborderDimensionTag)
	{
		return m_entity->coborderContainer(CoborderDimensionTag());
	}

	static void setEntityVertex(EntityType &entity, VertexLocalIndexType localIndex, VertexGlobalIndexType globalIndex)
	{
		entity.setVertex(localIndex, globalIndex);
	}

private:
	EntityType *m_entity;
	GlobalIndexType m_entityIndex;

	void initEntityBorder(MeshInitializerType &meshInitializer) { BaseType::initEntityBorder(*this, meshInitializer); }
	void initEntityCoborder()                                   { CoborderBaseType::initEntityCoborder(*this); }

	template<typename DimensionTag>
	class BorderEntitiesCreator
	{
		typedef EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag> BorderTag;
		typedef typename BorderTag::BorderEntityType                        BorderEntityType;
		typedef typename BorderTag::BorderEntityTag                         BorderEntityTag;
		typedef typename BorderTag::ContainerType::IndexType                LocalIndexType;

		enum { borderEntitiesCount       = BorderTag::count };
		enum { borderEntityVerticesCount = EntityBorderTag<MeshConfigTag, BorderEntityTag, DimTag<0> >::count };

		typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimTag<0> >::ContainerType BorderVertexContainerType;

	public:
		typedef StaticContainer<BorderEntityType, LocalIndexType, borderEntitiesCount> BorderContainerType;

		static void createBorderEntities(BorderContainerType &borderEntities, const EntityType &entity)
		{
			const BorderVertexContainerType &borderVertexContainer = entity.borderContainer(DimTag<0>());
			LOOP<LocalIndexType, borderEntitiesCount, CreateBorderEntities>::EXEC(borderEntities, borderVertexContainer); // Static loop is necessary here because border entities are described using templates.
		}

	private:
		template<LocalIndexType borderEntityIndex>
		class CreateBorderEntities
		{
		public:
			static void exec(BorderContainerType &borderEntities, const BorderVertexContainerType &borderVertexContainer)
			{
				BorderEntityType &borderEntity = borderEntities[borderEntityIndex];
				LOOP<LocalIndexType, borderEntityVerticesCount, SetBorderEntityVertex>::EXEC(borderEntity, borderVertexContainer); // Static loop is necessary here because border entities are described using templates.
			}

		private:
			template<LocalIndexType borderEntityVertexIndex>
			class SetBorderEntityVertex
			{
			public:
				static void exec(BorderEntityType &borderEntity, const BorderVertexContainerType &borderVertexContainer)
				{
					LocalIndexType vertexIndex = BorderTag::template Vertex<borderEntityIndex, borderEntityVertexIndex>::index;
					EntityInitializer<MeshConfigTag, BorderEntityTag>::setEntityVertex(borderEntity, borderEntityVertexIndex, borderVertexContainer[vertexIndex]);
				}
			};
		};
	};
};


template<typename MeshConfigTag>
class EntityInitializer<MeshConfigTag, topology::Vertex> : public EntityInitializerCoborderLayer<MeshConfigTag, topology::Vertex, typename MeshTag<MeshConfigTag>::MeshDimTag>
{
	typedef DimTag<0> DimensionTag;

	typedef EntityInitializerCoborderLayer<MeshConfigTag, topology::Vertex, typename MeshTag<MeshConfigTag>::MeshDimTag> CoborderBaseType;

	typedef typename EntityTag<MeshConfigTag, DimensionTag>::Type                     EntityType;
	typedef typename EntityTag<MeshConfigTag, DimensionTag>::ContainerType::IndexType GlobalIndexType;

	typedef MeshInitializer<MeshConfigTag> MeshInitializerType;

public:
	EntityInitializer() : m_entity(0), m_entityIndex(-1) {}

	void init(EntityType &entity, GlobalIndexType entityIndex)
	{
		m_entity = &entity;
		m_entityIndex = entityIndex;
	}

	void initEntity(MeshInitializerType &meshInitializer)
	{
		m_entity->setMesh(meshInitializer.getMesh());
		m_entity->setID(m_entityIndex);

		initEntityCoborder();
	}

	template<typename CoborderDimensionTag>
	typename EntityCoborderTag<MeshConfigTag, topology::Vertex, CoborderDimensionTag>::ContainerType &getEntityCoborderContainer(CoborderDimensionTag)
	{
		return m_entity->coborderContainer(CoborderDimensionTag());
	}

private:
	EntityType *m_entity;
	GlobalIndexType m_entityIndex;

	void initEntityCoborder() { CoborderBaseType::initEntityCoborder(*this); }
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<true>, StorageTag<true> > : public EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
	typedef EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous> BaseType;

	typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::ContainerType ContainerType;
	typedef typename ContainerType::DataType                                                    GlobalIndexType;

	typedef MeshInitializer<MeshConfigTag>                  MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

protected:
	void initEntityBorder(EntityInitializerType &entityInitializer, MeshInitializerType &meshInitializer)
	{
		typedef typename EntityInitializerType::template BorderEntities<DimensionTag>::ContainerType BorderEntitiesContainerType; // This cannot be on the class level because EntityInitializerType is inherited from EntityInitializerLayer

		BorderEntitiesContainerType borderEntities;
		entityInitializer.template createBorderEntities<DimensionTag>(borderEntities);

		ContainerType &borderContainer = entityInitializer.getEntityBorderContainer(DimensionTag());
		for (typename BorderEntitiesContainerType::IndexType i = 0; i < borderEntities.size(); i++)
		{
			GlobalIndexType borderEntityIndex = meshInitializer.findEntityIndex(borderEntities[i]);
			GlobalIndexType coborderEntityIndex = entityInitializer.getEntityIndex();
			borderContainer[i] = borderEntityIndex;
			meshInitializer.getEntityInitializer(DimensionTag(), borderEntityIndex).addCoborderEntity(DimTag<MeshEntityTag::dimension>(), coborderEntityIndex);
		}

		BaseType::initEntityBorder(entityInitializer, meshInitializer);
	}
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<true>, StorageTag<false> > : public EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
	typedef EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous> BaseType;

	typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::ContainerType ContainerType;

	typedef MeshInitializer<MeshConfigTag>                  MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

protected:
	void initEntityBorder(EntityInitializerType &entityInitializer, MeshInitializerType &meshInitializer)
	{
		typedef typename EntityInitializerType::template BorderEntities<DimensionTag>::ContainerType BorderEntitiesContainerType; // This cannot be on the class level because EntityInitializerType is inherited from EntityInitializerLayer

		BorderEntitiesContainerType borderEntities;
		entityInitializer.template createBorderEntities<DimensionTag>(borderEntities);

		ContainerType &borderContainer = entityInitializer.getEntityBorderContainer(DimensionTag());
		for (typename BorderEntitiesContainerType::IndexType i = 0; i < borderEntities.size(); i++)
			borderContainer[i] = meshInitializer.findEntityIndex(borderEntities[i]);

		BaseType::initEntityBorder(entityInitializer, meshInitializer);
	}
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<false>, StorageTag<true> > : public EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
	typedef EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous> BaseType;

	typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::ContainerType::DataType GlobalIndexType;

	typedef MeshInitializer<MeshConfigTag>                  MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

protected:
	void initEntityBorder(EntityInitializerType &entityInitializer, MeshInitializerType &meshInitializer)
	{
		typedef typename EntityInitializerType::template BorderEntities<DimensionTag>::ContainerType BorderEntitiesContainerType; // This cannot be on the class level because EntityInitializerType is inherited from EntityInitializerLayer

		BorderEntitiesContainerType borderEntities;
		entityInitializer.template createBorderEntities<DimensionTag>(borderEntities);

		for (typename BorderEntitiesContainerType::IndexType i = 0; i < borderEntities.size(); i++)
		{
			GlobalIndexType borderEntityIndex = meshInitializer.findEntityIndex(borderEntities[i]);
			GlobalIndexType coborderEntityIndex = entityInitializer.getEntityIndex();
			meshInitializer.getEntityInitializer(DimensionTag(), borderEntityIndex).addCoborderEntity(DimTag<MeshEntityTag::dimension>(), coborderEntityIndex);
		}

		BaseType::initEntityBorder(entityInitializer, meshInitializer);
	}
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<false>, StorageTag<false> > : public EntityInitializerLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
};


template<typename MeshConfigTag, typename MeshEntityTag>
class EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimTag<0>, StorageTag<true>, StorageTag<true> >
{
	typedef DimTag<0> DimensionTag;

	typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::ContainerType ContainerType;
	typedef typename ContainerType::DataType                                                    GlobalIndexType;

	typedef MeshInitializer<MeshConfigTag>                  MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

protected:
	void initEntityBorder(EntityInitializerType &entityInitializer, MeshInitializerType &meshInitializer)
	{
		const ContainerType &borderContainer = entityInitializer.getEntityBorderContainer(DimensionTag());
		for (typename ContainerType::IndexType i = 0; i < borderContainer.size(); i++)
		{
			GlobalIndexType borderEntityIndex = borderContainer[i];
			GlobalIndexType coborderEntityIndex = entityInitializer.getEntityIndex();
			meshInitializer.getEntityInitializer(DimensionTag(), borderEntityIndex).addCoborderEntity(DimTag<MeshEntityTag::dimension>(), coborderEntityIndex);
		}
	}
};


template<typename MeshConfigTag, typename MeshEntityTag>
class EntityInitializerLayer<MeshConfigTag, MeshEntityTag, DimTag<0>, StorageTag<true>, StorageTag<false> >
{
	typedef MeshInitializer<MeshConfigTag>                  MeshInitializerType;
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

protected:
	void initEntityBorder(EntityInitializerType &, MeshInitializerType &) {}
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<true> > : public EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
	typedef EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous> BaseType;

	typedef typename EntityCoborderTag<MeshConfigTag, MeshEntityTag, DimensionTag>::GrowableContainerType GrowableContainerType;
	typedef typename GrowableContainerType::DataType                                                      GlobalIndexType;

	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

public:
	using BaseType::addCoborderEntity;
	void addCoborderEntity(DimensionTag, GlobalIndexType entityIndex) { m_coborderContainer.insert(entityIndex); }

protected:
	void initEntityCoborder(EntityInitializerType &entityInitializer)
	{
		m_coborderContainer.copy(entityInitializer.getEntityCoborderContainer(DimensionTag()));
		m_coborderContainer.free();

		BaseType::initEntityCoborder(entityInitializer);
	}

private:
	GrowableContainerType m_coborderContainer;
};


template<typename MeshConfigTag, typename MeshEntityTag, typename DimensionTag>
class EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, DimensionTag, StorageTag<false> > : public EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, typename DimensionTag::Previous>
{
};


template<typename MeshConfigTag, typename MeshEntityTag>
class EntityInitializerCoborderLayer<MeshConfigTag, MeshEntityTag, DimTag<MeshEntityTag::dimension>, StorageTag<false> >
{
	typedef EntityInitializer<MeshConfigTag, MeshEntityTag> EntityInitializerType;

protected:
	void addCoborderEntity()                         {} // This method is due to 'using BaseType::...;' in the derived classes.
	void initEntityCoborder(EntityInitializerType &) {}
};


} // namespace implementation


#endif
