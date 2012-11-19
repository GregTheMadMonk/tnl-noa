#if !defined(_CONFIG_TETRAHEDRAL_H_)
#define _CONFIG_TETRAHEDRAL_H_

#include <mesh/topology/tetrahedron.h>
#include <mesh/config/config_base.h>


namespace config
{


class TetrahedralMeshConfig : public MeshConfigBase
{
public:
	typedef topology::Tetrahedron Cell;

	enum { dimWorld = Cell::dimension };
};


} // namespace config


using config::TetrahedralMeshConfig;


#endif
