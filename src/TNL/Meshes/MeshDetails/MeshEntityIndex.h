/***************************************************************************
                          MeshEntityIndex.h  -  description
                             -------------------
    begin                : Feb 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Meshes {

template< typename IDType >
class MeshEntityIndex
{
public:
   // FIXME: IDType may be unsigned
   MeshEntityIndex()
      : id( -1 )
   {}

   __cuda_callable__
   const IDType& getIndex() const
   {
      TNL_ASSERT_GE( this->id, 0, "entity index is negative" );
      return this->id;
   }

   __cuda_callable__
   bool operator==( const MeshEntityIndex& id ) const
   {
      return ( this->id == id.id );
   }

protected:
   __cuda_callable__
   void setIndex( IDType id )
   {
      this->id = id;
   }

   IDType id;
};

template<>
class MeshEntityIndex< void >
{
public:
   __cuda_callable__
   bool operator==( const MeshEntityIndex& id ) const
   {
      return true;
   }

protected:
   template< typename Index >
   __cuda_callable__
   void setIndex( Index )
   {}
};

} // namespace Meshes
} // namespace TNL