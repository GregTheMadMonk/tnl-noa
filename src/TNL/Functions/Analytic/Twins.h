/***************************************************************************
                          ExpBump.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimension >
class TwinsBase : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );
};

template< int Dimension,
          typename Real >
class Twins
{
};

template< typename Real >
class Twins< 1, Real > : public TwinsBase< Real, 1 >
{
   public:

      enum { Dimension = 1 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      static String getType();

      Twins();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder,
                typename Point >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class Twins< 2, Real > : public TwinsBase< Real, 2 >
{
   public:

      enum { Dimension = 2 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      static String getType();

      Twins();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder,
                typename Point >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class Twins< 3, Real > : public TwinsBase< Real, 3 >
{
   public:

      enum { Dimension = 3 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      static String getType();

      Twins();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder,
                typename Point >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const Twins< Dimension, Real >& f )
{
   str << "Twins function.";
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/Twins_impl.h>

