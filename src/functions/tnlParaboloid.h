/***************************************************************************
                          tnlParaboloid.h  -  description
                             -------------------
    begin                : Oct 13, 2014
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

#pragma once

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlDomain.h>

template< int dimensions,
          typename Real = double >
class tnlParaboloidBase : public tnlDomain< dimensions, SpaceDomain >
{
   public:

   tnlParaboloidBase();

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setXCentre( const Real& waveLength );

   Real getXCentre() const;

   void setYCentre( const Real& waveLength );

   Real getYCentre() const;

   void setZCentre( const Real& waveLength );

   Real getZCentre() const;

   void setCoefficient( const Real& coefficient );

   Real getCoefficient() const;

   void setOffset( const Real& offset );

   Real getOffset() const;

   protected:

   Real xCentre, yCentre, zCentre, coefficient, radius;
};

template< int Dimensions, typename Real >
class tnlParaboloid
{
};

template< typename Real >
class tnlParaboloid< 1, Real > : public tnlParaboloidBase< 1, Real >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 1, RealType > VertexType;

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
class tnlParaboloid< 2, Real > : public tnlParaboloidBase< 2, Real >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 2, RealType > VertexType;

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
class tnlParaboloid< 3, Real > : public tnlParaboloidBase< 3, Real >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 3, RealType > VertexType;



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
ostream& operator << ( ostream& str, const tnlParaboloid< Dimensions, Real >& f )
{
   str << "SDF Paraboloid function: amplitude = " << f.getCoefficient()
       << " offset = " << f.getOffset();
   return str;
}

#include <functions/tnlParaboloid_impl.h>

