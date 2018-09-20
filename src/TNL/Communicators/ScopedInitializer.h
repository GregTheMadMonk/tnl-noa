/***************************************************************************
                          ScopedInitializer.h  -  description
                             -------------------
    begin                : Sep 16, 2018
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

namespace TNL {
namespace Communicators {

template< typename Communicator >
struct ScopedInitializer
{
   ScopedInitializer( int& argc, char**& argv )
   {
      Communicator::Init( argc, argv );
   }

   ~ScopedInitializer()
   {
      Communicator::Finalize();
   }
};

} // namespace Communicators
} // namespace TNL
