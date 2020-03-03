/***************************************************************************
                          ConfigEntryList.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <TNL/Config/ConfigEntry.h>

namespace TNL {
namespace Config {

template< typename EntryType >
class ConfigEntryList : public ConfigEntry< EntryType, std::vector< EntryType > >
{
public:
   // inherit constructors
   using ConfigEntry< EntryType, std::vector< EntryType > >::ConfigEntry;
};

} // namespace Config
} // namespace TNL
