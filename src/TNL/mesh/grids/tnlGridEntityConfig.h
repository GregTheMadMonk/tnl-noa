/***************************************************************************
                          tnlGridEntityConfig.h  -  description
                             -------------------
    begin                : Dec 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

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

class tnlGridEntityNoStencilStorage
{
   public:
 
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( int neighbourEntityStorage )
      {
         return tnlGridEntityNoStencil;
      }
 
      constexpr static int getStencilSize()
      {
         return 0;
      }
};

template< int stencilSize = 1 >
class tnlGridEntityCrossStencilStorage
{
   public:
 
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( const int neighbourEntityDimensions )
      {
         //return tnlGridEntityNoStencil;
         return ( GridEntity::entityDimensions == GridEntity::GridType::meshDimensions &&
                  neighbourEntityDimensions == GridEntity::GridType::meshDimensions )
                * tnlGridEntityCrossStencil;
      }
 
      constexpr static int getStencilSize()
      {
         return stencilSize;
      }
};

} // namespace TNL

