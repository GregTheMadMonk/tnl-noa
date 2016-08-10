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
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimensions >
class PseudoSquareBase : public Domain< Dimensions, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

   protected:

      RealType height;
};

template< int Dimensions,
          typename Real >
class PseudoSquare
{
};

template< typename Real >
class PseudoSquare< 1, Real > : public PseudoSquareBase< Real, 1 >
{
   public:

      enum { Dimensions = 1 };
      typedef Real RealType;
      typedef Vectors::StaticVector< Dimensions, Real > VertexType;

      static String getType();

      PseudoSquare();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const;
};

template< typename Real >
class PseudoSquare< 2, Real > : public PseudoSquareBase< Real, 2 >
{
   public:

      enum { Dimensions = 2 };
      typedef Real RealType;
      typedef Vectors::StaticVector< Dimensions, Real > VertexType;

      static String getType();

      PseudoSquare();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const;
};

template< typename Real >
class PseudoSquare< 3, Real > : public PseudoSquareBase< Real, 3 >
{
   public:

      enum { Dimensions = 3 };
      typedef Real RealType;
      typedef Vectors::StaticVector< Dimensions, Real > VertexType;

      static String getType();

      PseudoSquare();

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const PseudoSquare< Dimensions, Real >& f )
{
   str << "Level-set pseudo square function.";
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namepsace TNL

#include <TNL/Functions/Analytic/PseudoSquare_impl.h>
