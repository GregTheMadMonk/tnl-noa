/***************************************************************************
                          Traits.h  -  description
                             -------------------
    begin                : Jun 4, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes {

template< typename T >
class isGrid
: public std::false_type
{};

template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
class isGrid< Grid< Dimension, Real, Device, Index > >
: public std::true_type
{};

} // namespace Meshes
} // namespace TNL
