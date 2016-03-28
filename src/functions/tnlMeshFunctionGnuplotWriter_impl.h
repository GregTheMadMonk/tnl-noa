/***************************************************************************
                          tnlMeshFunctionGnuplotWriter.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by oberhuber
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

#ifndef TNLMESHFUNCTIONGNUPLOTWRITER_IMPL_H
#define	TNLMESHFUNCTIONGNUPLOTWRITER_IMPL_H

template< typename MeshFunction >
bool
tnlMeshFunctionGnuplotWriter< MeshFunction >::
write( const MeshFunction& function,
       ostream& str )
{
   std::cerr << "Gnuplot writer for mesh functions defined on mesh type " << MeshFunction::MeshType::getType() << " is not (yet) implemented." << std::endl;
   return false;   
}

/****
 * 1D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       ostream& str )
{ 
   const MeshType& mesh = function.getMesh();
   typename MeshType::Cell entity( mesh );
   for( entity.getCoordinates().x() = 0;
        entity.getCoordinates().x() < mesh.getDimensions().x();
        entity.getCoordinates().x() ++ )
   {
      entity.refresh();
      typename MeshType::VertexType v = entity.getCenter();
      str << v << " " 
          << function.getData().getElement( entity.getIndex() ) << std::endl;
   }
   return true;
}

/****
 * 1D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       ostream& str )
{ 
   const MeshType& mesh = function.getMesh();
   typename MeshType::Vertex entity( mesh );
   for( entity.getCoordinates().x() = 0;
        entity.getCoordinates().x() <= mesh.getDimensions().x();
        entity.getCoordinates().x() ++ )
   {
      entity.refresh();
      typename MeshType::VertexType v = entity.getCenter();
      str << v << " " 
          << function.getData().getElement( entity.getIndex() ) << std::endl;      
   }
   return true;
}


/****
 * 2D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 2, Real > >::
write( const MeshFunctionType& function,
       ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Cell entity( mesh );
   for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() < mesh.getDimensions().y();
        entity.getCoordinates().y() ++ ) 
   {
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;      
      }
      str << std::endl;
   }
   return true;
}

/****
 * 2D grid, faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typedef typename MeshType::Face EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientation;
   EntityType entity( mesh );
   
   entity.setOrientation( EntityOrientation( 1.0, 0.0 ) );
   for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() < mesh.getDimensions().y();
        entity.getCoordinates().y() ++ )       
   {
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() <= mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;      
      }
      str << std::endl;
   }
   
   entity.setOrientation( EntityOrientation( 0.0, 1.0 ) );
         for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )

   {
            for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() <= mesh.getDimensions().y();
        entity.getCoordinates().y() ++ ) 

      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;      
      }
      str << std::endl;
   }   
   return true;
}


/****
 * 2D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Vertex entity( mesh );
   for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() <= mesh.getDimensions().y();
        entity.getCoordinates().y() ++ )   
   {
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() <= mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;      
      }
      str << std::endl;
   }
   return true;
}

#endif	/* TNLMESHFUNCTIONGNUPLOTWRITER_IMPL_H */
