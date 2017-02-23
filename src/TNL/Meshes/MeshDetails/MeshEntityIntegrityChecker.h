/***************************************************************************
                          MeshEntityIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {

template< typename MeshEntity >
class MeshEntityIntegrityChecker
{
   public:

      static bool checkEntity( const MeshEntity& entity )
      {
         return true;
      }

};

} // namespace Meshes
} // namespace TNL
