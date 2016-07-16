/***************************************************************************
                          tnlDynamicTypeTag.h  -  description
                             -------------------
    begin                : Mar 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLDYNAMICTYPETAG_H_
#define TNLDYNAMICTYPETAG_H_

template< typename Element >
struct tnlDynamicTypeTag
{
   enum { value = false };
};


#endif /* TNLDYNAMICTYPETAG_H_ */
