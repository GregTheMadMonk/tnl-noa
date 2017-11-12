/***************************************************************************
                          GridEntityGetter.h  -  description
                             -------------------
    begin                : Nov 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {

template< typename Grid,
          typename GridEntity,
          int EntityDimension = GridEntity::getEntityDimension() >
class GridEntityGetter
{
   //static_assert( false, "Wrong mesh type or entity topology." );
};

/***
 * The main code is in template specializations in GridEntityIndexer.h
 */

} // namespace Meshes
} // namespace TNL

