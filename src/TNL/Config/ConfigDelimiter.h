/***************************************************************************
                          ConfigDelimiter.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <TNL/Config/ConfigEntryBase.h>

namespace TNL {
namespace Config {

class ConfigDelimiter : public ConfigEntryBase
{
public:
   ConfigDelimiter( const std::string& delimiter )
   : ConfigEntryBase( "", delimiter, false )
   {}

   virtual bool isDelimiter() const override { return true; };

   virtual std::string getUIEntryType() const override { return ""; };
};

} //namespace Config
} //namespace TNL
