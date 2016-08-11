/***************************************************************************
                          GridEntityConfig.h  -  description
                             -------------------
    begin                : Dec 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {

enum GridEntityStencilStorage
{ 
   GridEntityNoStencil = 0,
   GridEntityCrossStencil,
   GridEntityFullStencil
};

/****
 * This class says what neighbour grid entity indexes shall be pre-computed and stored in the
 * grid entity structure. If neighbourEntityStorage() returns false, nothing is stored.
 * Otherwise, if neighbour entity storage is enabled, we may store either only neighbour entities in a cross like this
 *
 *                X
 *   X            X
 *  XOX    or   XXOXX   etc.
 *   X            X
 *                X
 *
 * or all neighbour entities like this
 *
 *           XXXXX
 *  XXX      XXXXX
 *  XOX  or  XXOXX  etc.
 *  XXX      XXXXX
 *           XXXXX
 */

class GridEntityNoStencilStorage
{
   public:
 
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( int neighbourEntityStorage )
      {
         return GridEntityNoStencil;
      }
 
      constexpr static int getStencilSize()
      {
         return 0;
      }
};

template< int stencilSize = 1 >
class GridEntityCrossStencilStorage
{
   public:
 
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( const int neighbourEntityDimensions )
      {
         //return GridEntityNoStencil;
         return ( GridEntity::entityDimensions == GridEntity::GridType::meshDimensions &&
                  neighbourEntityDimensions == GridEntity::GridType::meshDimensions )
                * GridEntityCrossStencil;
      }
 
      constexpr static int getStencilSize()
      {
         return stencilSize;
      }
};

} // namespace Meshes
} // namespace TNL

