/***************************************************************************
                          Grid.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>

namespace TNL {
namespace Meshes {

template< int Dimension,
          typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Grid;

template< int Dimension, typename Real, typename Device, typename Index >
bool operator==( const Grid< Dimension, Real, Device, Index >& lhs,
                 const Grid< Dimension, Real, Device, Index >& rhs )
{
   return lhs.getDimensions() == rhs.getDimensions()
       && lhs.getOrigin() == rhs.getOrigin()
       && lhs.getProportions() == rhs.getProportions();
}

template< int Dimension, typename Real, typename Device, typename Index >
bool operator!=( const Grid< Dimension, Real, Device, Index >& lhs,
                 const Grid< Dimension, Real, Device, Index >& rhs )
{
   return ! (lhs == rhs);
}

template< int Dimension, typename Real, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const Grid< Dimension, Real, Device, Index >& grid )
{
   str << "Grid dimensions:    " << grid.getDimensions()  << std::endl;
   str << "     origin:        " << grid.getOrigin()      << std::endl;
   str << "     proportions:   " << grid.getProportions() << std::endl;
   str << "     localBegin:    " << grid.getLocalBegin() << std::endl;
   str << "     localEnd:      " << grid.getLocalEnd() << std::endl;
   str << "     interiorBegin: " << grid.getInteriorBegin() << std::endl;
   str << "     interiorEnd:   " << grid.getInteriorEnd() << std::endl;
   return str;
}

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
