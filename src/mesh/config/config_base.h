#if !defined(_MESH_CONFIG_BASE_H_)
#define _MESH_CONFIG_BASE_H_

#include <mesh/common/common.h>


namespace config
{


class MeshConfigBase
{
public:
	typedef double          NumericType;
	typedef int             GlobalIndexType;
	typedef int             LocalIndexType;
	typedef GlobalIndexType IDType; // Set to void to disable storage of entity ID
};


// Explicit storage of all mesh entities by default
template<typename MeshConfigTag, DimensionType dimension>
struct EntityStorage { enum { enabled = true }; };

// By default, all border entities of a mesh entity are stored provided that they are stored in the mesh
template<typename MeshConfigTag, typename MeshEntityTag, DimensionType dimension>
struct BorderStorage { enum { enabled = EntityStorage<MeshConfigTag, dimension>::enabled }; };

// By default, no coborder entities of any mesh entity are stored
template<typename MeshConfigTag, typename MeshEntityTag, DimensionType dimension>
struct CoborderStorage { enum { enabled = false }; };


} // namespace config


#endif
