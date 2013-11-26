/***************************************************************************
                          tnlFunctionDiscretizer.h  -  description
                             -------------------
    begin                : Nov 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLFUNCTIONDISCRETIZER_H_
#define TNLFUNCTIONDISCRETIZER_H_

template< typename Mesh, typename Function, typename DiscreteFunction >
class tnlFunctionDiscretizer
{
   public:

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   static void discretize( const Mesh& mesh,
                           const Function& function,
                           DiscreteFunction& discreteFunction );
};

#include <implementation/generators/functions/tnlFunctionDiscretizer_impl.h>

#endif /* TNLFUNCTIONDISCRETIZER_H_ */
