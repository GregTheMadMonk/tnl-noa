/***************************************************************************
                          tnlSDFSign.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSDFSIGN_H_
#define TNLSDFSIGN_H_

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

template< typename Real = double >
class tnlSDFSignBase
{
   public:

   tnlSDFSignBase();


   void setEpsilon( const Real& epsilon );

   Real getEpsilon() const;

   protected:

   Real epsilon;
};

template< typename Mesh, int Dimensions, typename Real, typename FunctionType, int SignMod >
class tnlSDFSign
{
};

template< typename Mesh, typename Real, typename FunctionType, int SignMod >
class tnlSDFSign< Mesh, 1, Real, FunctionType, SignMod > : public tnlSDFSignBase< Real >
{
   public:

   enum { Dimensions = 1 };
   typedef Containers::StaticVector< 1, Real > VertexType;
   typedef Real RealType;

   bool setup( const Config::ParameterContainer& parameters,
           const String& prefix = ""  );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#endif

   protected:

   FunctionType function;

};

template< typename Mesh, typename Real, typename FunctionType, int SignMod >
class tnlSDFSign< Mesh, 2, Real, FunctionType, SignMod > : public tnlSDFSignBase< Real >
{
   public:

   enum { Dimensions = 2 };
   typedef Containers::StaticVector< 2, Real > VertexType;
   typedef Real RealType;

   bool setup( const Config::ParameterContainer& parameters,
           const String& prefix = ""  );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#endif

   protected:

   FunctionType function;

};

template< typename Mesh, typename Real, typename FunctionType, int SignMod >
class tnlSDFSign< Mesh, 3, Real, FunctionType, SignMod > : public tnlSDFSignBase< Real >
{
   public:

   enum { Dimensions = 3 };
   typedef Containers::StaticVector< 3, Real > VertexType;
   typedef Real RealType;

   bool setup( const Config::ParameterContainer& parameters,
           const String& prefix = ""  );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#endif

   protected:

   FunctionType function;

};

#include <functions/tnlSDFSign_impl.h>

#endif /* TNLSDFSIGN_H_ */

