/***************************************************************************
                          MeshEntityId.h  -  description
                             -------------------
    begin                : Feb 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

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

   const IDType &getId() const
   {
      TNL_ASSERT( this->id >= 0, );
      return this->id;
   }

   void setId( GlobalIndexType id )
   {
      this->id = id;
   }

   bool operator==( const MeshEntityId< IDType, GlobalIndexType >& id ) const
   {
      return ( this->id == id.id );
   }

   protected:
   IDType id;
};

template< typename GlobalIndexType >
class MeshEntityId< void, GlobalIndexType >
{
   public:
   void setId( GlobalIndexType )
   {}

   bool operator==( const MeshEntityId< void, GlobalIndexType >& id ) const
   {
      return true;
   }
};

} // namespace Meshes
} // namespace TNL
