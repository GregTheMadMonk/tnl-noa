#if !defined(_CONFIG_HEXAHEDRAL_H_)
#define _CONFIG_HEXAHEDRAL_H_

#include <mesh/topology/hexahedron.h>
#include <mesh/config/config_base.h>


namespace config
{


class HexahedralMeshConfig : public MeshConfigBase
{
public:
	typedef topology::Hexahedron Cell;

	enum { dimension = Cell::dimension };
	enum { dimWorld = dimension };
};


} // namespace config


using config::HexahedralMeshConfig;


#endif
