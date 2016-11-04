/***************************************************************************
                          MeshEntityId.h  -  description
                             -------------------
    begin                : Feb 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Assert.h>

namespace TNL {
namespace Meshes {

template< typename IDType,
          typename GlobalIndexType >
class MeshEntityId
{
public:
   MeshEntityId()
      : id( -1 )
   {}

   const IDType& getId() const
   {
      TNL_ASSERT( this->id >= 0, );
      return this->id;
   }

   bool operator==( const MeshEntityId< IDType, GlobalIndexType >& id ) const
   {
      return ( this->id == id.id );
   }

protected:
   void setId( GlobalIndexType id )
   {
      this->id = id;
   }

   IDType id;
};

template< typename GlobalIndexType >
class MeshEntityId< void, GlobalIndexType >
{
public:
   bool operator==( const MeshEntityId< void, GlobalIndexType >& id ) const
   {
      return true;
   }

protected:
   void setId( GlobalIndexType )
   {}
};

} // namespace Meshes
} // namespace TNL
