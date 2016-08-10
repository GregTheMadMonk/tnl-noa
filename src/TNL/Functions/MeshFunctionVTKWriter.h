/***************************************************************************
                          MeshFunctionVTKWriter.h  -  description
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
class MeshFunctionVTKWriter
{
   public:
 
      static bool write( const MeshFunction& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunction& function,
                         std::ostream& str ){}
};

/***
 * 1D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 1, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};
 
/***
 * 1D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 0, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 2D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 2, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 2D grid, faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 1, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 2D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 0, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 3D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 3, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 3D grid, faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 2, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 3D grid, edges
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1, Real > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 1, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

/***
 * 3D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshFunctionVTKWriter< MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::MeshFunction< MeshType, 0, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         std::ostream& str );
      static void writeHeader(const MeshFunctionType& function,
                         std::ostream& str );
};

} // namespace Functions
} // namespace TNL

