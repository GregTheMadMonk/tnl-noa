/***************************************************************************
                          tnlConfigDelimiter.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLCONFIGDELIMITER_H_
#define TNLCONFIGDELIMITER_H_

struct tnlConfigDelimiter : public tnlConfigEntryBase
{
   tnlConfigDelimiter( const tnlString& delimiter )
   : tnlConfigEntryBase( "", delimiter, false )
   {
   };

   bool isDelimiter() const { return true; };

   tnlString getEntryType() const { return ""; };

   tnlString getUIEntryType() const { return ""; };
};

#endif /* TNLCONFIGDELIMITER_H_ */
