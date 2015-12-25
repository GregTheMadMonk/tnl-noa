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

#include <functions/tnlFunction.h>

template< int Dimensions >
class tnlExactLinearDiffusion
{};

template<>
class tnlExactLinearDiffusion< 1 > : public tnlFunction< 1, AnalyticFunction >
{
   public:

      static tnlString getType();
   
      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType getValue( const Function& function,
                                            const typename Function::VertexType& v,
                                            const typename Function::RealType& time = 0.0 ) const;
};

template<>
class tnlExactLinearDiffusion< 2 > : public tnlFunction< 2, AnalyticFunction >
{
   public:
      
      static tnlString getType();

      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType getValue( const Function& function,
                                            const typename Function::VertexType& v,
                                            const typename Function::RealType& time = 0.0 ) const;
};

template<>
class tnlExactLinearDiffusion< 3 > : public tnlFunction< 3 >
{
   public:
      
      static tnlString getType();

      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType getValue( const Function& function,
                                            const typename Function::VertexType& v,
                                            const typename Function::RealType& time = 0.0 ) const;
};

#include <operators/diffusion/tnlExactLinearDiffusion_impl.h>

#endif /* TNLEXACTLINEARDIFFUSION_H_ */
