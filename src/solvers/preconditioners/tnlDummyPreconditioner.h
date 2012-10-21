/***************************************************************************
                          tnlDummyPreconditioner.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLDUMMYPRECONDITIONER_H_
#define TNLDUMMYPRECONDITIONER_H_

#include <core/tnlObject.h>

template< typename Real, tnlDevice Device, typename Index >
class tnlDummyPreconditioner
{
   public:

   template< typename Vector >
      bool solve( const Vector& b, Vector& x ) {};
};


#endif /* TNLDUMMYPRECONDITIONER_H_ */
