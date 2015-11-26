/***************************************************************************
                          tnlFunction.h  -  description
                             -------------------
    begin                : Nov 8, 2015
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


#ifndef TNLFUNCTION_H
#define	TNLFUNCTION_H

enum tnlFunctionType { tnlGeneralFunction, 
                       tnlDiscreteFunction,
                       tnlAnalyticFunction };

template< tnlFunctionType FunctionType >
class tnlFunction
{
   public:
   
      //static constexpr tnlFunctionType getFunctionType() { return FunctionType; }
      enum { functionType = FunctionType };
};

#endif	/* TNLFUNCTION_H */

