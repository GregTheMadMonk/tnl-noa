#if !defined(_MESH_INFO_H_)
#define _MESH_INFO_H_

#include <iostream>
#include <mesh/global/static_if.h>
#include <mesh/global/static_loop.h>
#include <mesh/config.h>
#include <mesh/traversal.h>


template<typename MeshEntityType>
class EntityInfo
{
	enum { meshDimension = MeshEntityType::MeshConfig::dimension };
	enum { entityDimension = MeshEntityType::Tag::dimension };

public:
	static void print(std::ostream &stream, const MeshEntityType &entity)
	{
		printCoords(stream, entity);
		printEntityBorders(stream, entity);
		printEntityCoborders(stream, entity);
		printMemoryConsumption(stream, entity);
	}

private:
	class PrintVertexCoords
	{
	public:
		static void exec(std::ostream &stream, const MeshEntityType &entity)
		{
			typedef typename MeshEntityType::PointType PointType;

			stream << " coordinates: ";

			const PointType &point = entity.getPoint();
			for (DimensionType d = 0; d < PointType::dimension; d++)
				stream << point[d] << " ";
		}
	};

	template<DimensionType dimension>
	class EntityBorderPrinter
	{
	public:
		static void exec(std::ostream &stream, const MeshEntityType &entity)
		{
			IF<BorderEntitiesAvailable<MeshEntityType, dimension>::value, PrintBorderEntities>::EXEC(stream, entity);
		}

	private:
		class PrintBorderEntities
		{
		public:
			static void exec(std::ostream &stream, const MeshEntityType &entity)
			{
				typedef typename BorderRange<MeshEntityType, dimension>::Type BorderRange;
				typedef typename BorderRange::DataType                        BorderEntityType;
				typedef typename BorderRange::IndexType                       IndexType;

				stream << " border" << dimension << ": ";
				BorderRange borderRange = borderEntities<dimension>(entity);
				for (IndexType i = 0; i < borderRange.size(); i++)
				{
					const BorderEntityType &borderEntity = borderRange[i];
					stream << borderEntity.getID() << " ";
				}
			}
		};
	};

	template<DimensionType dimension>
	class EntityCoborderPrinter
	{
		enum { coborderDimension = dimension + entityDimension + 1 };

	public:
		static void exec(std::ostream &stream, const MeshEntityType &entity)
		{
			IF<CoborderEntitiesAvailable<MeshEntityType, coborderDimension>::value, PrintCoborderEntities>::EXEC(stream, entity);
		}

	private:
		class PrintCoborderEntities
		{
		public:
			static void exec(std::ostream &stream, const MeshEntityType &entity)
			{
				typedef typename CoborderRange<MeshEntityType, coborderDimension>::Type CoborderRange;
				typedef typename CoborderRange::DataType                                CoborderEntityType;
				typedef typename CoborderRange::IndexType                               IndexType;

				stream << " coborder" << coborderDimension << ": ";
				CoborderRange coborderRange = coborderEntities<coborderDimension>(entity);
				for (IndexType i = 0; i < coborderRange.size(); i++)
				{
					const CoborderEntityType &coborderEntity = coborderRange[i];
					stream << coborderEntity.getID() << " ";
				}
			}
		};
	};

	static void printCoords(std::ostream &stream, const MeshEntityType &entity)
	{
		IF<entityDimension == 0, PrintVertexCoords>::EXEC(stream, entity);
	}

	static void printEntityBorders(std::ostream &stream, const MeshEntityType &entity)
	{
		LOOP<DimensionType, entityDimension, EntityBorderPrinter>::EXEC(stream, entity);
	}

	static void printEntityCoborders(std::ostream &stream, const MeshEntityType &entity)
	{
		LOOP<DimensionType, meshDimension - entityDimension, EntityCoborderPrinter>::EXEC(stream, entity);
	}

	static void printMemoryConsumption(std::ostream &stream, const MeshEntityType &entity)
	{
		stream << " memory consumption: " << sizeof(entity) + entity.allocatedMemorySize() << " bytes";
	}
};


template<typename MeshType>
class MeshInfo
{
	enum { meshDimension = MeshType::Config::dimension };

public:
	static void print(std::ostream &stream, const MeshType &mesh)
	{
		printEntities(stream, mesh);
	}

private:
	template<DimensionType dimension>
	class MeshEntitiesPrinter
	{
	public:
		static void exec(std::ostream &stream, const MeshType &mesh)
		{
			IF<EntitiesAvailable<MeshType, dimension>::value, PrintEntities>::EXEC(stream, mesh);
		}

	private:
		class PrintEntities
		{
		public:
			static void exec(std::ostream &stream, const MeshType &mesh)
			{
				typedef typename EntityRange<MeshType, dimension>::Type EntityRange;
				typedef typename EntityRange::DataType                  EntityType;
				typedef typename EntityRange::IndexType                 IndexType;

				stream << std::endl << "Entities" << dimension << ":" << std::endl;
				EntityRange entityRange = entities<dimension>(mesh);
				for (IndexType i = 0; i < entityRange.size(); i++)
				{
					const EntityType &entity = entityRange[i];
					stream << "[" << i << "](" << entity.getID() << ")";
					EntityInfo<EntityType>::print(stream, entity);
					stream << std::endl;
				}
			}
		};
	};

	static void printEntities(std::ostream &stream, const MeshType &mesh)
	{
		LOOP<DimensionType, meshDimension + 1, MeshEntitiesPrinter>::EXEC(stream, mesh);
	}
};


#endif // !defined(_MESH_INFO_H_)
