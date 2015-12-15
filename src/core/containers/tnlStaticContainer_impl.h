/***************************************************************************
                          tnlStaticContainer_impl.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSTATICCONTAINER_IMPL_H_
#define TNLSTATICCONTAINER_IMPL_H_

template< int Size, typename Element >
tnlStaticContainer< Size, Element >::tnlStaticContainer()
{
}

template< int Size, typename Element >
tnlString tnlStaticContainer< Size, Element >::getType()
{
   return tnlString( "tnlStaticContainer< " ) +
          tnlString( Size ) +
          tnlString( ", " ) +
          ::getType< Element >() +
          tnlString( " >" );
}

template< int Size, typename Element >
int tnlStaticContainer< Size, Element >::getSize() const
{
   return size;
}

template< int Size, typename Element >
Element& tnlStaticContainer< Size, Element >::operator[]( const int id )
{
   return this->data[ id ];
}

template< int Size, typename Element >
const Element& tnlStaticContainer< Size, Element >::operator[]( const int id ) const
{
   return this->data[ id ];
}

template< int Size, typename Element >
Element tnlStaticContainer< Size, Element >::getElement( const int id ) const
{
   return this->data[ id ];
}

template< int Size, typename Element >
void tnlStaticContainer< Size, Element >::setElement( const int id,
                                                      const ElementType& data )
{
   this->data[ id ] = data;
}

template< int Size, typename Element >
bool tnlStaticContainer< Size, Element >::save( tnlFile& file ) const
{
   if( ! tnlObject::save( file ) ||
       ! this->data.save( file ) )
   {
      cerr << "I am no able to save " << this->getType() << "." << endl;
      return false;
   }
   return true;
}

template< int Size, typename Element >
bool tnlStaticContainer< Size, Element >::load( tnlFile& file )
{
   if( ! tnlObject::load( file ) ||
       ! this->data.load( file ) )
   {
      cerr << "I am no able to load " << this->getType() << "." << endl;
      return false;
   }
   return true;
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

// TODO: this does not work with CUDA 5.5 - fix it later

/*extern template class tnlStaticContainer< 1, char >;
extern template class tnlStaticContainer< 1, int >;
extern template class tnlStaticContainer< 1, float >;
extern template class tnlStaticContainer< 1, double >;

extern template class tnlStaticContainer< 2, char >;
extern template class tnlStaticContainer< 2, int >;
extern template class tnlStaticContainer< 2, float >;
extern template class tnlStaticContainer< 2, double >;

extern template class tnlStaticContainer< 3, char >;
extern template class tnlStaticContainer< 3, int >;
extern template class tnlStaticContainer< 3, float >;
extern template class tnlStaticContainer< 3, double >;

extern template class tnlStaticContainer< 4, char >;
extern template class tnlStaticContainer< 4, int >;
extern template class tnlStaticContainer< 4, float >;
extern template class tnlStaticContainer< 4, double >;
*/

#endif /* TEMPLATE_EXPLICIT_INSTANTIATION */


#endif /* TNLSTATICCONTAINER_IMPL_H_ */
