/***************************************************************************
                          tnlDummyProblem.h  -  description
                             -------------------
    begin                : Jul 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLDUMMYPROBLEM_H_
#define TNLDUMMYPROBLEM_H_

#include <core/tnlHost.h>
#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlDummyProblem
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlVector< Real, Device, Index > DofVectorType;
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
};



#endif /* TNLDUMMYPROBLEM_H_ */
