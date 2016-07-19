/***************************************************************************
                          tnlExpBumpFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/config/tnlParameterContainer.h>
#include <TNL/core/vectors/tnlStaticVector.h>
#include <TNL/functions/tnlDomain.h>
#include <TNL/core/tnlCuda.h>

namespace TNL {

template< typename Real,
          int Dimensions >
class tnlBlobFunctionBase : public tnlDomain< Dimensions, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

     protected:

      RealType height;
};

template< int Dimensions,
          typename Real >
class tnlBlobFunction
{
};

template< typename Real >
class tnlBlobFunction< 1, Real > : public tnlBlobFunctionBase< Real, 1 >
{
   public:

      enum { Dimensions = 1 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlBlobFunction();

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
class tnlBlobFunction< 2, Real > : public tnlBlobFunctionBase< Real, 2 >
{
   public:

      enum { Dimensions = 2 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlBlobFunction();

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
class tnlBlobFunction< 3, Real > : public tnlBlobFunctionBase< Real, 3 >
{
   public:

      enum { Dimensions = 3 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlBlobFunction();

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
std::ostream& operator << ( std::ostream& str, const tnlBlobFunction< Dimensions, Real >& f )
{
   str << "Level-set pseudo square function.";
   return str;
}

} // namepsace TNL

#include <TNL/functions/initial_conditions/level_set_functions/tnlBlobFunction_impl.h>

