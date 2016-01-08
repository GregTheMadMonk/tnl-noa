/***************************************************************************
                          tnlExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
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

#ifndef TNLCYLINDERFUNCTION_H_
#define TNLCYLINDERFUNCTION_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlDomain.h>
#include <core/tnlCuda.h>

template< typename Real,
          int Dimensions >
class tnlCylinderFunctionBase : public tnlDomain< Dimensions, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

      void setDiameter( const RealType& sigma );

      const RealType& getDiameter() const;

   protected:

      RealType diameter;
};

template< int Dimensions,
          typename Real >
class tnlCylinderFunction
{
};

template< typename Real >
class tnlCylinderFunction< 1, Real > : public tnlCylinderFunctionBase< Real, 1 >
{
   public:

      enum { Dimensions = 1 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlCylinderFunction();

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
      __cuda_callable__
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< typename Real >
class tnlCylinderFunction< 2, Real > : public tnlCylinderFunctionBase< Real, 2 >
{
   public:

      enum { Dimensions = 2 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlCylinderFunction();

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
      __cuda_callable__
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< typename Real >
class tnlCylinderFunction< 3, Real > : public tnlCylinderFunctionBase< Real, 3 >
{
   public:

      enum { Dimensions = 3 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlCylinderFunction();

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
      __cuda_callable__
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< int Dimensions,
          typename Real >
ostream& operator << ( ostream& str, const tnlCylinderFunction< Dimensions, Real >& f )
{
   str << "Cylinder function.";
   return str;
}

#include <functions/initial_conditions/tnlCylinderFunction_impl.h>


#endif /* TNLEXPBUMPFUNCTION_H_ */
