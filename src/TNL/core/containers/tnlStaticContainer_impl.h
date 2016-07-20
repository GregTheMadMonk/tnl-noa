/***************************************************************************
                          tnlStaticContainer_impl.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< int Size, typename Element >
tnlStaticContainer< Size, Element >::tnlStaticContainer()
{
}

template< int Size, typename Element >
String tnlStaticContainer< Size, Element >::getType()
{
   return String( "tnlStaticContainer< " ) +
          String( Size ) +
          String( ", " ) +
         TNL::getType< Element >() +
          String( " >" );
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
bool tnlStaticContainer< Size, Element >::save( File& file ) const
{
   if( ! Object::save( file ) ||
       ! this->data.save( file ) )
   {
      std::cerr << "I am no able to save " << this->getType() << "." << std::endl;
      return false;
   }
   return true;
}

template< int Size, typename Element >
bool tnlStaticContainer< Size, Element >::load( File& file )
{
   if( ! Object::load( file ) ||
       ! this->data.load( file ) )
   {
      std::cerr << "I am no able to load " << this->getType() << "." << std::endl;
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


} // namespace TNL
