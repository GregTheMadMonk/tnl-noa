#if !defined(_CONFIG_TRIANGULAR_NOEDGES_H_)
#define _TRIANGULAR_NOEDGES_H_

#include "triangular.h"


namespace config
{


class TriangularMeshNoEdgesConfig : public TriangularMeshConfig
{
};

template<> struct EntityStorage<TriangularMeshNoEdgesConfig, 1> { enum { enabled = false }; }; // Edges not stored


} // namespace config


using config::TriangularMeshNoEdgesConfig;


#endif
