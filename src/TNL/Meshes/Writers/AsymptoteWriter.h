/***************************************************************************
                          AsymptoteWriter.h  -  description
                             -------------------
    begin                : Oct 22, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
class AsymptoteWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The Asymptote format supports only 1D, 2D and 3D meshes." );

public:
   static void writeAllEntities( const Mesh& mesh, std::ostream& str )
   {
      throw Exceptions::NotImplementedError();
   }

   template< int EntityDimension = Mesh::getMeshDimension() >
   static void writeEntities( const Mesh& mesh, std::ostream& str )
   {
      throw Exceptions::NotImplementedError();
   }
};

template< typename Real, typename Device, typename Index >
class AsymptoteWriter< Grid< 2, Real, Device, Index > >
{
   using Mesh = Grid< 2, Real, Device, Index >;
   using CoordinatesType = typename Mesh::CoordinatesType;
   using PointType = typename Mesh::PointType;

public:
   static void writeAllEntities( const Mesh& mesh, std::ostream& str )
   {
      str << "size( "
          << mesh.getProportions(). x() << "cm , "
          << mesh.getProportions(). y() << "cm );"
          << std::endl << std::endl;
      typename Mesh::Vertex vertex( mesh );
      CoordinatesType& vertexCoordinates = vertex.getCoordinates();
      PointType v;
      for( Index j = 0; j < mesh.dimensions. y(); j ++ )
      {
         str << "draw( ";
         vertexCoordinates.x() = 0;
         vertexCoordinates.y() = j;
         v = vertex.getCenter();
         str << "( " << v. x() << ", " << v. y() << " )";
         for( Index i = 0; i < mesh.dimensions. x(); i ++ )
         {
            vertexCoordinates.x() = i + 1;
            vertexCoordinates.y() = j;
            v = vertex.getCenter();
            str << "--( " << v. x() << ", " << v. y() << " )";
         }
         str << " );" << std::endl;
      }
      str << std::endl;
      for( Index i = 0; i < mesh.dimensions. x(); i ++ )
      {
         str << "draw( ";
         vertexCoordinates.x() = i;
         vertexCoordinates.y() = 0;
         v = vertex.getCenter();
         str << "( " << v. x() << ", " << v. y() << " )";
         for( Index j = 0; j < mesh.dimensions. y(); j ++ )
         {
            vertexCoordinates.x() = i;
            vertexCoordinates.y() = j + 1;
            v = vertex.getCenter();
            str << "--( " << v. x() << ", " << v. y() << " )";
         }
         str << " );" << std::endl;
      }
      str << std::endl;

      typename Mesh::Cell cell( mesh );
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      const Real cellMeasure = mesh.getSpaceSteps().x() * mesh.getSpaceSteps().y();
      for( Index i = 0; i < mesh.dimensions. x(); i ++ )
         for( Index j = 0; j < mesh.dimensions. y(); j ++ )
         {
            cellCoordinates.x() = i;
            cellCoordinates.y() = j;
            v = vertex.getCenter();
            str << "label( scale(0.33) * Label( \"$" << std::setprecision( 3 ) << cellMeasure << std::setprecision( 8 )
                << "$\" ), ( " << v. x() << ", " << v. y() << " ), S );" << std::endl;
         }

      for( Index i = 0; i < mesh.dimensions. x(); i ++ )
         for( Index j = 0; j < mesh.dimensions. y(); j ++ )
         {
            PointType v1, v2, c;

            /****
             * East edge normal
             */
            /*v1 = mesh.getPoint( CoordinatesType( i + 1, j ), v1 );
            v2 = mesh.getPoint( CoordinatesType( i + 1, j + 1 ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            mesh.getEdgeNormal< 1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            str << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" << std::endl;
            */
            /****
             * West edge normal
             */
            /*mesh.getPoint< -1, -1 >( CoordinatesType( i, j ), v1 );
            mesh.getPoint< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            mesh.getEdgeNormal< -1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            str << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" << std::endl;
            */
            /****
             * North edge normal
             */
            /*mesh.getPoint< 1, 1 >( CoordinatesType( i, j ), v1 );
            mesh.getPoint< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            mesh.getEdgeNormal< 0, 1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            str << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" << std::endl;
            */
            /****
             * South edge normal
             */
            /*mesh.getPoint< 1, -1 >( CoordinatesType( i, j ), v1 );
            mesh.getPoint< -1, -1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            mesh.getEdgeNormal< 0, -1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            str << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" << std::endl;
            */
         }
   }

   template< int EntityDimension = Mesh::getMeshDimension() >
   static void writeEntities( const Mesh& mesh, std::ostream& str )
   {
      throw Exceptions::NotImplementedError();
   }
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL
