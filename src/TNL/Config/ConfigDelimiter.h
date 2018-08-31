/***************************************************************************
                          ConfigDelimiter.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {
namespace Config {

struct ConfigDelimiter : public ConfigEntryBase
{
   ConfigDelimiter( const String& delimiter )
   : ConfigEntryBase( "", delimiter, false )
   {
   };

   bool isDelimiter() const { return true; };

   String getEntryType() const { return ""; };

   String getUIEntryType() const { return ""; };
   
   //~ConfigDelimiter(){};
};

} //namespace Config
} //namespace TNL
