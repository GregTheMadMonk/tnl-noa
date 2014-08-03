/***************************************************************************
                          tnlTestFunction.h  -  description
                             -------------------
    begin                : Aug 2, 2014
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

#ifndef TNLTESTFUNCTION_H_
#define TNLTESTFUNCTION_H_


template< int FunctionDimensions,
          typename Real,
          typename Device >
class tnlTestingFunction
{
   protected:

   enum TestFunctions{ none,
                       constant,
                       expBump,
                       sinBumps,
                       sinWaves };

   public:

   enum{ Dimensions = FunctionDimensions };
   typedef Real RealType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;

   tnlTestingFunction();

   bool init( const tnlParameterContainer& parameters );

   template< typename Vertex,
             typename Real = typename Vertex::RealType >
   Real getValue( const Vertex& vertex ) const;

   ~tnlTestingFunction();

   protected:

   void* function;

   TestFunction functionType;

};

#include <implementation/functions/tnlTestFunction_impl.h>

#endif /* TNLTESTFUNCTION_H_ */
