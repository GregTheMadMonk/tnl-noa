/***************************************************************************
                          tnlExactLinearDiffusion.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLEXACTLINEARDIFFUSION_H_
#define TNLEXACTLINEARDIFFUSION_H_

template< int Dimensions >
class tnlExactLinearDiffusion
{};

template<>
class tnlExactLinearDiffusion< 1 >
{
   public:

      enum { Dimensions = 1 };

      static tnlString getType();
   
#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif      
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0 );
};

template<>
class tnlExactLinearDiffusion< 2 >
{
   public:

      enum { Dimensions = 2 };

      static tnlString getType();

#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif      
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0 );
};

template<>
class tnlExactLinearDiffusion< 3 >
{
   public:

      enum { Dimensions = 3 };

      static tnlString getType();

#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif      
      static Real getValue( const Function& function,
                            const Vertex& v,
                            const Real& time = 0.0 );
};

#include <implementation/operators/diffusion/tnlExactLinearDiffusion_impl.h>

#endif /* TNLEXACTLINEARDIFFUSION_H_ */
