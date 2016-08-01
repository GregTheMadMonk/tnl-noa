/***************************************************************************
                          ExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Functions {   

template< typename Real,
          int Dimensions >
class CylinderFunctionBase : public Domain< Dimensions, SpaceDomain >
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
class CylinderFunction
{
};

template< typename Real >
class CylinderFunction< 1, Real > : public CylinderFunctionBase< Real, 1 >
{
   public:

      enum { Dimensions = 1 };
      typedef Real RealType;
      typedef Vectors::StaticVector< Dimensions, Real > VertexType;

      static String getType();

      CylinderFunction();

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
class CylinderFunction< 2, Real > : public CylinderFunctionBase< Real, 2 >
{
   public:

      enum { Dimensions = 2 };
      typedef Real RealType;
      typedef Vectors::StaticVector< Dimensions, Real > VertexType;

      static String getType();

      CylinderFunction();

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
class CylinderFunction< 3, Real > : public CylinderFunctionBase< Real, 3 >
{
   public:

      enum { Dimensions = 3 };
      typedef Real RealType;
      typedef Vectors::StaticVector< Dimensions, Real > VertexType;

      static String getType();

      CylinderFunction();

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
std::ostream& operator << ( std::ostream& str, const CylinderFunction< Dimensions, Real >& f )
{
   str << "Cylinder function.";
   return str;
}

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/CylinderFunction_impl.h>

