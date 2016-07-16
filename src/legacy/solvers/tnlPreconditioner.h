/***************************************************************************
                          tnlPreconditioner.h  -  description
                             -------------------
    begin                : 2007/02/01
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlPreconditionerH
#define tnlPreconditionerH

template< typename T > class tnlPreconditioner
{
   public:

   virtual ~tnlPreconditioner() {};

   virtual bool Solve( const T* b, T* x ) const = 0;
};

#endif
