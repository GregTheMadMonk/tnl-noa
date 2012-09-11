#if !defined(_MESH_INFORMATION_H_)
#define _MESH_INFORMATION_H_

#include "common/common.h"


template<typename MeshType, DimensionType dimension>
class EntitiesAvailable
{
public:
	enum { value = config::EntityStorage<typename MeshType::Config, dimension>::enabled };
};


template<typename EntityType, DimensionType dimension>
class BorderEntitiesAvailable
{
public:
	enum { value = config::BorderStorage<typename EntityType::MeshConfig, typename EntityType::Tag, dimension>::enabled };
};


template<typename EntityType, DimensionType dimension>
class CoborderEntitiesAvailable
{
public:
	enum { value = config::CoborderStorage<typename EntityType::MeshConfig, typename EntityType::Tag, dimension>::enabled };
};


#endif
