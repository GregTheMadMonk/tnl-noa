/***************************************************************************
                          PseudoSquare.h  -  description
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
class PseudoSquareBase : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

   protected:

      RealType height;
};

template< int Dimension,
          typename Real >
class PseudoSquare
{
};

template< typename Real >
class PseudoSquare< 1, Real > : public PseudoSquareBase< Real, 1 >
{
   public:

      enum { Dimension = 1 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

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
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
};

template< typename Real >
class PseudoSquare< 2, Real > : public PseudoSquareBase< Real, 2 >
{
   public:

      enum { Dimension = 2 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

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
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
};

template< typename Real >
class PseudoSquare< 3, Real > : public PseudoSquareBase< Real, 3 >
{
   public:

      enum { Dimension = 3 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

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
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const PseudoSquare< Dimension, Real >& f )
{
   str << "Level-set pseudo square function.";
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namepsace TNL

#include <TNL/Functions/Analytic/PseudoSquare_impl.h>
