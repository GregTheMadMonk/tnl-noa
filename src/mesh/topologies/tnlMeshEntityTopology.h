/***************************************************************************
                          tnlMeshEntityTopology.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHENTITYTOPOLOGY_H_
#define TNLMESHENTITYTOPOLOGY_H_

template< typename MeshEntityTopology,
          int SubentityDimension >
struct tnlMeshSubtopology
{
};

template< typename MeshEntityTopology,
          typename SubentityTopology,
          int SubentityIndex,
          int SubentityVertexIndex >
struct tnlSubentityVertex;


template< typename MeshConfig,
          int Dimensions >
class tnlMeshEntityTopology
{
   public:

   typedef typename tnlMeshSubtopology< typename MeshConfig::CellTopology,
                                        Dimensions >::Topology Topology;
};

template< typename MeshConfig >
class tnlMeshEntityTopology< MeshConfig,
                             MeshConfig::CellTopology::dimensions >
{
   public:

   typedef typename MeshConfig::CellTopology Topology;
};
#endif /* TNLMESHENTITYTOPOLOGY_H_ */
