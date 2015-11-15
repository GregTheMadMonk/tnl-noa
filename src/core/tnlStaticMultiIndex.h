/***************************************************************************
                          tnlStaticMultiIndex.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLSTATICMULTIINDEX_H
#define	TNLSTATICMULTIINDEX_H

template< int i1_ >
class tnlStaticMultiIndex1D
{
   public:
      
      static const int i1 = i1_;
};

template< int i1_,
          int i2_ >
class tnlStaticMultiIndex2D
{
   public:
      
      static const int i1 = i1_;
      
      static const int i2 = i2_;
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
};




#endif	/* TNLSTATICMULTIINDEX_H */

