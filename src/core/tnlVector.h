/***************************************************************************
                          tnlVector.h  -  description
                             -------------------
    begin                : Oct 3, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLLONGVECTOR_H_
#define TNLLONGVECTOR_H_

#include <core/tnlArray.h>

template< typename RealType, typename Device = tnlHost, typename IndexType = int >
class tnlVector : public tnlArray< RealType, Device, IndexType >
{

};

#include <core/tnlVectorHost.h>
#include <core/tnlVectorCUDA.h>

#endif /* TNLLONGVECTOR_H_ */
