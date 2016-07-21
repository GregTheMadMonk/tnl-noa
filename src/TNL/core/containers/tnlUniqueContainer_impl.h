/***************************************************************************
                          tnlUniqueContainer_impl.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Element, typename Key >
class tnlUniqueContainer : public Object
   class ConstIterator
   {

template< typename Element, typename Key >
const Element& tnlUniqueContainer< Element, Key >::ConstIterator::operator*() const
{

}

template< typename Element, typename Key >
const Element* tnlUniqueContainer< Element, Key >::ConstIterator::operator->() const
{

}

template< typename Element, typename Key >
   typename tnlUniqueContainer< Element, Key >::ConstIterator&
      tnlUniqueContainer< Element, Key >::ConstIterator::operator++()
{

}

template< typename Element, typename Key >
typename tnlUniqueContainer< Element, Key >::ConstIterator
   tnlUniqueContainer< Element, Key >::ConstIterator::operator++( int )
{

}

template< typename Element, typename Key >
bool tnlUniqueContainer< Element, Key >::ConstIterator::operator==( const ConstIterator& it ) const
{

}

template< typename Element, typename Key >
bool tnlUniqueContainer< Element, Key >::ConstIterator::operator!=( const ConstIterator& it ) const
{

}

template< typename Element, typename Key >
typename tnlUniqueContainer< Element, Key >::ConstIterator
   tnlUniqueContainer< Element, Key >::insertElement( const ElementType& e );

template< typename Element, typename Key >
typename tnlUniqueContainer< Element, Key >::ConstIterator
   tnlUniqueContainer< Element, Key >::findElement( const ElementType& e ) const;

template< typename Element, typename Key >
size_t tnlUniqueContainer< Element, Key >::getSize() const;

template< typename Element, typename Key >
void tnlUniqueContainer< Element, Key >::reset();

template< typename Element, typename Key >
bool tnlUniqueContainer< Element, Key >::save( File& file ) const;

template< typename Element, typename Key >
bool tnlUniqueContainer< Element, Key >::load( File& file );


} // namespace TNL
