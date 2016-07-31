/***************************************************************************
                          tnlGnuplotWriter.h  -  description
                             -------------------
    begin                : Jul 2, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <TNL/Vectors/StaticVector.h>

namespace TNL {

class tnlGnuplotWriter
{
   public:

      template< typename Element >
      static void write( std::ostream& str,
                         const Element& d )
      {
         str << d;
      }

      template< typename Real >
      static void write( std::ostream& str,
                         const Vectors::StaticVector< 1, Real >& d )
      {
         str << d.x() << " ";
      }

      template< typename Real >
      static void write( std::ostream& str,
                         const Vectors::StaticVector< 2, Real >& d )
      {
         str << d.x() << " " << d.y() << " ";
      }

      template< typename Real >
      static void write( std::ostream& str,
                         const Vectors::StaticVector< 3, Real >& d )
      {
         str << d.x() << " " << d.y() << " " << d. z() << " ";
      }

};

} // namespace TNL
