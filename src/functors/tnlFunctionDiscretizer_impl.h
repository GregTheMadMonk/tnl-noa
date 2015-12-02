/***************************************************************************
                          tnlFunctionDiscretizer_impl.h  -  description
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

#ifndef TNLFUNCTIONDISCRETIZER_IMPL_H_
#define TNLFUNCTIONDISCRETIZER_IMPL_H_

template< typename Mesh, typename Function, typename Vector >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
void tnlFunctionDiscretizer< Mesh, Function, Vector >::discretize( const Mesh& mesh,
                                                                   const Function& function,
                                                                   Vector& discreteFunction,
                                                                   const typename Vector::RealType& time )
{
   //tnlAssert( Mesh::Dimensions == Function::Dimensions, ); // TODO: change this to tnlStaticAssert
   IndexType i = 0;
   discreteFunction.setSize( mesh.template getEntitiesCount< Mesh::Cells >() );
   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      while( i < mesh.template getEntitiesCount< Mesh::Cells >() )
      {
         VertexType v = mesh.template getEntity< Mesh::Cells >( i ).getCenter();
         discreteFunction[ i ] = function.template getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( v, time );
         i++;
      }
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      // TODO: implement
   }
}



#endif /* TNLFUNCTIONDISCRETIZER_IMPL_H_ */
