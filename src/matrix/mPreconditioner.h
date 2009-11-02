/***************************************************************************
                          mPreconditioner.h  -  description
                             -------------------
    begin                : 2007/02/01
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mPreconditionerH
#define mPreconditionerH

template< typename T > class mPreconditioner
{
   public:

   virtual ~mPreconditioner() {};

   virtual bool Solve( const T* b, T* x ) const = 0;
};

#endif
