/***************************************************************************
                          tnlGridEntityGetter.h  -  description
                             -------------------
    begin                : Nov 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLGRIDENTITYGETTER_H
#define TNLGRIDENTITYGETTER_H

template< typename Grid,
          typename GridEntity,
          int EntityDimensions = GridEntity::entityDimensions >
class tnlGridEntityGetter
{
   //static_assert( false, "Wrong mesh type or entity topology." );
};

/***
 * The main code is in template specializations in tnlGridEntityIndexer.h
 */

#endif	/* TNLGRIDENTITYGETTER_H */

