#if !defined(_MESH_ENTITY_KEY_H_)
#define _MESH_ENTITY_KEY_H_

#include "mesh_tags.h"
#include "mesh_entity.h"


namespace implementation
{


// Unique identification of a mesh entity by its vertices. Uniqueness is preserved for entities of the same type only.
template<typename MeshConfigTag, typename MeshEntityTag>
class MeshEntityKey
{
	typedef MeshEntity<MeshConfigTag, MeshEntityTag> MeshEntityType;

	typedef typename EntityBorderTag<MeshConfigTag, MeshEntityTag, DimTag<0> >::ContainerType ContainerType;

public:
	explicit MeshEntityKey(const MeshEntityType &entity)
	{
		m_vertexIDs = entity.borderContainer(DimTag<0>());
		m_vertexIDs.sort();
	}

	bool operator<(const MeshEntityKey &other) const
	{
		for (typename ContainerType::IndexType i = 0; i < m_vertexIDs.size(); i++)
		{
			if (m_vertexIDs[i] < other.m_vertexIDs[i])
				return true;
			else if (m_vertexIDs[i] > other.m_vertexIDs[i])
				return false;
		}

		return false;
	}

private:
	ContainerType m_vertexIDs;
};


} // namespace implementation


#endif
