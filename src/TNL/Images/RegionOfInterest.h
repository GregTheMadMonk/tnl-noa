// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Images/Image.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Images {

template< typename Index = int >
class RegionOfInterest
{
public:
   RegionOfInterest();

   bool
   setup( const Config::ParameterContainer& parameters, const Image< Index >* image );

   bool
   check( const Image< Index >* image ) const;

   Index
   getTop() const;

   Index
   getBottom() const;

   Index
   getLeft() const;

   Index
   getRight() const;

   Index
   getWidth() const;

   Index
   getHeight() const;

   template< typename Grid >
   bool
   setGrid( Grid& grid, bool verbose = false );

   bool
   isIn( Index row, Index column ) const;

protected:
   Index top, bottom, left, right;
};

}  // namespace Images
}  // namespace TNL

#include <TNL/Images/RegionOfInterest_impl.h>
