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

template< typename Nonlinearity, int Dimensions >
class tnlExactNonlinearDiffusion
{};

template< typename Nonlinearity >
class tnlExactNonlinearDiffusion< Nonlinearity, 1 > 
   : public tnlDomain< 1, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactNonlinearDiffusion< " + Nonlinearity::getType() + ", 1 >";
      };
      
      void setNonlinearity( const Nonlinearity& nonlinearity )
      {
         this->nonlinearity = nonlinearity;
      }
   
      template< typename Function >
      __cuda_callable__
      Real operator()( const Function& function,
                       const Vertex& v,
                       const Real& time = 0.0 ) const
      {         
         const Real u_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const Real u_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
         const Real g = nonlinearity( function, v, time ) 
         const Real g_x = nonlinearity::template getPartialDerivative< Function, 1, 0, 0 >( function, v, time );
         return u_xx - u_x * g_x / g;          
      }
   
      protected:
         
         Nonlinearity nonlinearity;
};

template< typename Nonlinearity >
class tnlExactNonlinearDiffusion< Nonlinearity, 2 >
   : public tnlDomain< 2, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactNonlinearDiffusion< " + Nonlinearity::getType() + ", 2 >";
      };

      void setNonlinearity( const Nonlinearity& nonlinearity )
      {
         this->nonlinearity = nonlinearity;
      }

      template< typename Function >
      __cuda_callable__
      Real operator()( const Function& function,
                       const VertexType& v,
                       const Real& time = 0.0 ) const;

      protected:
         
         Nonlinearity nonlinearity;
      
};

template< typename Nonlinearity >
class tnlExactNonlinearDiffusion< Nonlinearity, 3 >
   : public tnlDomain< 3, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactNonlinearDiffusion< " + Nonlinearity::getType() + ", 3 >";
      }

      void setNonlinearity( const Nonlinearity& nonlinearity )
      {
         this->nonlinearity = nonlinearity;
      }      
      
      template< typename Function >
      __cuda_callable__
      Real operator()( const Function& function,
                       const VertexType& v,
                       const Real& time = 0.0 ) const;
      
      protected:
         
         Nonlinearity nonlinearity;
};

#include "tnlExactNonlinearDiffusion_impl.h"

#endif /* TNLEXACTNONLINEARDIFFUSION_H_ */
