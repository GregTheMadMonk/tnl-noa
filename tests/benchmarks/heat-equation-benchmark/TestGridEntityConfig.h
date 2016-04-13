/***************************************************************************
                          TestGridEntityConfig.h  -  description
                             -------------------
    begin                : Dec 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once

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

class TestGridEntityNoStencilStorage
{
   public:
      
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( int neighbourEntityStorage )
      {
         return TestGridEntityNoStencil;
      }
      
      constexpr static int getStencilSize()
      {
         return 0;
      }
};

template< int stencilSize = 1 >
class TestGridEntityCrossStencilStorage
{
   public:
      
      template< typename GridEntity >
      constexpr static bool neighbourEntityStorage( const int neighbourEntityDimensions )
      {
         //return TestGridEntityNoStencil;
         return ( GridEntity::entityDimensions == GridEntity::GridType::meshDimensions &&
                  neighbourEntityDimensions == GridEntity::GridType::meshDimensions )
                * TestGridEntityCrossStencil;         
      }
            
      constexpr static int getStencilSize()
      {
         return stencilSize;
      }
};
