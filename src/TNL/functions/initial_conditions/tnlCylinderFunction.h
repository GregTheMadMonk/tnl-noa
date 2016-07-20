/***************************************************************************
                          tnlExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/functions/tnlDomain.h>
#include <TNL/core/tnlCuda.h>

namespace TNL {

template< typename Real,
          int Dimensions >
class tnlCylinderFunctionBase : public tnlDomain< Dimensions, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

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
      typedef Vectors::tnlStaticVector< Dimensions, Real > VertexType;

      static String getType();

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
      RealType getPartialDerivative( const Vertex& v,
                                     const Real& time = 0.0 ) const;

      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class tnlCylinderFunction< 2, Real > : public tnlCylinderFunctionBase< Real, 2 >
{
   public:

      enum { Dimensions = 2 };
      typedef Real RealType;
      typedef Vectors::tnlStaticVector< Dimensions, Real > VertexType;

      static String getType();

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
      RealType getPartialDerivative( const Vertex& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class tnlCylinderFunction< 3, Real > : public tnlCylinderFunctionBase< Real, 3 >
{
   public:

      enum { Dimensions = 3 };
      typedef Real RealType;
      typedef Vectors::tnlStaticVector< Dimensions, Real > VertexType;

      static String getType();

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
      RealType getPartialDerivative( const Vertex& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const tnlCylinderFunction< Dimensions, Real >& f )
{
   str << "Cylinder function.";
   return str;
}

} // namespace TNL

#include <TNL/functions/initial_conditions/tnlCylinderFunction_impl.h>

