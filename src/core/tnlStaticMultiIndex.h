/***************************************************************************
                          tnlStaticMultiIndex.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< int i1_ >
class tnlStaticMultiIndex1D
{
   public:
 
      static const int i1 = i1_;
 
      static const int size = 1;
};

template< int i1_,
          int i2_ >
class tnlStaticMultiIndex2D
{
   public:
 
      static const int i1 = i1_;
 
      static const int i2 = i2_;
 
      static const int size = 2;
};

template< int i1_,
          int i2_,
          int i3_ >
class tnlStaticMultiIndex3D
{
   public:
 
      static const int i1 = i1_;
 
      static const int i2 = i2_;
 
      static const int i3 = i3_;
 
      static const int size = 3;
};

} // namespace TNL

