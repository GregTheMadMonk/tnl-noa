/***************************************************************************
                          tnlFunctionDiscretizer.h  -  description
                             -------------------
    begin                : Nov 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLFUNCTIONDISCRETIZER_H_
#define TNLFUNCTIONDISCRETIZER_H_

template< typename Mesh, typename Function, typename Vector >
class tnlFunctionDiscretizer
{
   public:

      typedef typename Vector::DeviceType DeviceType;
      typedef typename Mesh::IndexType IndexType;
      typedef typename Mesh::VertexType VertexType;
      typedef typename Mesh::CoordinatesType CoordinatesType;

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
   static void discretize( const Mesh& mesh,
                           const Function& function,
                           Vector& discreteFunction,
                           const typename Vector::RealType& time = 0 );
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   static void discretize( const Mesh& mesh,
                           const Function& function,
                           Vector& discreteFunction,
                           const typename Vector::RealType& time = 0 );
#endif   
   
};

#include <functors/tnlFunctionDiscretizer_impl.h>

#endif /* TNLFUNCTIONDISCRETIZER_H_ */
