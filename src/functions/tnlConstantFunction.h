/***************************************************************************
                          tnlConstantFunction.h  -  description
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

#ifndef TNLCONSTANTFUNCTION_H_
#define TNLCONSTANTFUNCTION_H_

#include <core/vectors/tnlStaticVector.h>

template< int FunctionDimensions,
          typename Real >
class tnlConstantFunction
{
   public:

   enum { Dimensions = FunctionDimensions };
   typedef Real RealType;
   typedef tnlStaticVector< Dimensions, Real > VertexType;

   tnlConstantFunction();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setValue( const RealType& value );

   const RealType& getValue() const;

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
#endif
   RealType getValue( const Vertex& v,
                      const Real& time = 0.0 ) const;

   protected:

   RealType value;
};

#include <implementation/functions/tnlConstantFunction_impl.h>

#endif /* TNLCONSTANTFUNCTION_H_ */
