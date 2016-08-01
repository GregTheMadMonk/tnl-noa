/***************************************************************************
                          tnlExactLinearDiffusion.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/Domain.h>

namespace TNL {

template< int Dimensions >
class tnlExactLinearDiffusion
{};

template<>
class tnlExactLinearDiffusion< 1 > : public Functions::Domain< 1, Functions::SpaceDomain >
{
   public:

      static const int Dimensions = 1;
 
      static String getType();
 
      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::VertexType& v,
                                              const typename Function::RealType& time = 0.0 ) const;
};

template<>
class tnlExactLinearDiffusion< 2 > : public Functions::Domain< 2, Functions::SpaceDomain >
{
   public:
 
      static const int Dimensions = 2;
 
      static String getType();

      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::VertexType& v,
                                              const typename Function::RealType& time = 0.0 ) const;
};

template<>
class tnlExactLinearDiffusion< 3 > : public Functions::Domain< 3 >
{
   public:
 
      static const int Dimensions = 3;
 
      static String getType();

      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::VertexType& v,
                                              const typename Function::RealType& time = 0.0 ) const;
};

} // namespace TNL

#include <TNL/operators/diffusion/tnlExactLinearDiffusion_impl.h>
