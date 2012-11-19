#if !defined(_CONFIG_LINEAR_H_)
#define _CONFIG_LINEAR_H_

#include <mesh/topology/edge.h>
#include <mesh/config/config_base.h>


namespace config
{


class LinearMeshConfig : public MeshConfigBase
{
public:
	typedef topology::Edge Cell;

	enum { dimWorld = Cell::dimension };
};


} // namespace config


using config::LinearMeshConfig;


#endif
