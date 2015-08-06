/***************************************************************************
                          tnlGnuplotWriter.h  -  description
                             -------------------
    begin                : Jul 2, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLGNUPLOTWRITER_H_
#define TNLGNUPLOTWRITER_H_

#include <ostream>
#include <core/vectors/tnlStaticVector.h>

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
                         const tnlStaticVector< 1, Real >& d )
      {
         str << d.x() << " ";
      }

      template< typename Real >
      static void write( std::ostream& str,
                         const tnlStaticVector< 2, Real >& d )
      {
         str << d.x() << " " << d.y() << " ";
      }

      template< typename Real >
      static void write( std::ostream& str,
                         const tnlStaticVector< 3, Real >& d )
      {
         str << d.x() << " " << d.y() << " " << d. z() << " ";
      }

};


#endif /* TNLGNUPLOTWRITER_H_ */
