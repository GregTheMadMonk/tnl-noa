/***************************************************************************
                          tnlSinWaveFunction.h  -  description
                             -------------------
    begin                : Nov 19, 2013
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

#ifndef TNLSDFSCHEMETEST_H_
#define TNLSDFSCHEMETEST_H_

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <functions/tnlSDFSinWaveFunction.h>
#include <functions/tnlSDFSinWaveFunctionSDF.h>
#include <functions/tnlSDFSinBumpsFunction.h>
#include <functions/tnlSDFSinBumpsFunctionSDF.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlSDFParaboloid.h>
#include <functions/tnlSDFParaboloidSDF.h>

template< typename function, typename Real = double >
class tnlSDFSchemeTestBase
{
   public:

   tnlSDFSchemeTestBase();

   bool setup( const Config::ParameterContainer& parameters,
           const String& prefix = "" );


   	function f;
};

template< typename function, int Dimensions, typename Real >
class tnlSDFSchemeTest
{

};

template< typename function, int Dimensions, typename Real >
class tnlSDFSchemeTest< function, 1, Real > : public tnlSDFSchemeTestBase< function, Real >
{
   public:


   enum { Dimensions = 1 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getValue( const VertexType& v,
           const Real& time = 0.0 ) const;



};

template< typename function, int Dimensions, typename Real >
class tnlSDFSchemeTest< function, 2, Real > : public tnlSDFSchemeTestBase< function, Real >
{
   public:


   enum { Dimensions = 2 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getValue( const VertexType& v,
           const Real& time = 0.0 ) const;


};

template< typename function, int Dimensions, typename Real >
class tnlSDFSchemeTest< function, 3, Real > : public tnlSDFSchemeTestBase< function,  Real >
{
   public:


   enum { Dimensions = 3 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getValue( const VertexType& v,
           const Real& time = 0.0 ) const;

};

#include <functions/tnlSDFSchemeTest_impl.h>

#endif /* TNLSDFSCHEMETEST_H_ */
