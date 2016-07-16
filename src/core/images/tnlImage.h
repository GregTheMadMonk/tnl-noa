/***************************************************************************
                          tnlImage.h  -  description
                             -------------------
    begin                : Jul 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLIMAGE_H
#define	TNLIMAGE_H

template< typename Index = int >
class tnlImage
{
   public:
 
      typedef Index IndexType;
 
      tnlImage() : width( 0 ), height( 0 ) {};
 
      IndexType getWidth() const
      {
         return this->width;
      }
 
      IndexType getHeight() const
      {
         return this->height;
      }
 
   protected:
 
      IndexType width, height;
 
};


#endif	/* TNLIMAGE_H */

