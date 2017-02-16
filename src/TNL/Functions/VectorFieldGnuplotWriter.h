/***************************************************************************
                          VectorFieldGnuplotWriter.h  -  description
                             -------------------
    begin                : Feb 16, 2017
    copyright            : (C) 2017 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Functions {

template< int, typename > class VectorField;

template< typename VectorField >
class VectorFieldGnuplotWriter
{
   public:

      static bool write( const VectorField& function,
                         std::ostream& str );
};

/***
 * 1D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > > >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};

/***
 * 1D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > > >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};


/***
 * 2D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 2, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};

/***
 * 2D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};

/***
 * 2D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > > >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< MeshType, 0, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};


/***
 * 3D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< MeshType, 3, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};

/***
 * 3D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< MeshType, 2, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};

/***
 * 3D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > > >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Functions::VectorField< MeshType, 0, RealType > > VectorFieldType;

      static bool write( const VectorFieldType& function,
                         std::ostream& str );
};

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/VectorFieldGnuplotWriter_impl.h>
