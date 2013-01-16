/***************************************************************************
                          tnlPreconditioner.h  -  description
                             -------------------
    begin                : 2007/02/01
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlPreconditionerH
#define tnlPreconditionerH

template< typename T > class tnlPreconditioner
{
   public:

   virtual ~tnlPreconditioner() {};

   virtual bool Solve( const T* b, T* x ) const = 0;
};

#endif
