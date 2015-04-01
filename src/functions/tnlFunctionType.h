/***************************************************************************
                          tnlFunctionType.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLFUNCTIONTYPE_H_
#define TNLFUNCTIONTYPE_H_

enum tnlFunctionTypeEnum { tnlGeneralFunction, tnlDiscreteFunction, tnlAnalyticFunction };

template< typename Function >
class tnlFunctionType
{
   public:

      enum { Type = tnlGeneralFunction };
};

#endif /* TNLFUNCTIONTYPE_H_ */
