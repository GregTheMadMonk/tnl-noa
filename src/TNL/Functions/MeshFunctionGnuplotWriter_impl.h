/***************************************************************************
                          MeshFunctionGnuplotWriter.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Functions {    

template< typename MeshFunction >
bool
MeshFunctionGnuplotWriter< MeshFunction >::
write( const MeshFunction& function,
       std::ostream& str )
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
MeshFunctionGnuplotWriter< MeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str )
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
MeshFunctionGnuplotWriter< MeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str )
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
MeshFunctionGnuplotWriter< MeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 2, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str )
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
MeshFunctionGnuplotWriter< MeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str )
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
MeshFunctionGnuplotWriter< MeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const MeshFunctionType& function,
       std::ostream& str )
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

} // namespace Functions
} // namespace TNL

