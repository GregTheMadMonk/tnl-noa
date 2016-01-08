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

#ifndef TNLEXACTNONLINEARDIFFUSION_H_
#define TNLEXACTNONLINEARDIFFUSION_H_

#include <functions/tnlDomain.h>

template< typename OperatorQ, int Dimensions >
class tnlExactNonlinearDiffusion
{};

template< typename OperatorQ >
class tnlExactNonlinearDiffusion< OperatorQ, 1 > : public tnlDomain< 1, SpaceDomain >
{
   public:

      enum { Dimensions = 1 };

      static tnlString getType();
   
#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif      
      
      __cuda_callable__
      Real operator()( const Function& function,
                       const Vertex& v,
                       const Real& time = 0.0 ) const;
};

template< typename OperatorQ >
class tnlExactNonlinearDiffusion< OperatorQ, 2 > : public tnlDomain< 2, SpaceDomain >
{
   public:

      enum { Dimensions = 2 };

      static tnlString getType();

#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif 

      __cuda_callable__
      Real operator()( const Function& function,
                       const Vertex& v,
                       const Real& time = 0.0 ) const;
};

template< typename OperatorQ >
class tnlExactNonlinearDiffusion< OperatorQ, 3 > : public tnlDomain< 3, SpaceDomain >
{
   public:

      enum { Dimensions = 3 };

      static tnlString getType();

#ifdef HAVE_NOT_CXX11      
      template< typename Function, typename Vertex, typename Real >
#else   
      template< typename Function, typename Vertex, typename Real = typename Vertex::RealType >
#endif 

      __cuda_callable__
      Real operator()( const Function& function,
                       const Vertex& v,
                       const Real& time = 0.0 ) const;
};

#include "tnlExactNonlinearDiffusion_impl.h"

#endif /* TNLEXACTNONLINEARDIFFUSION_H_ */
