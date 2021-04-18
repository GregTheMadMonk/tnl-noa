/***************************************************************************
                          SegmentsPrinting.h -  description
                             -------------------
    begin                : Apr 1, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/Containers/Array.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

/**
 * \brief Print segments sizes, i.e. the segments setup.
 *
 * \tparam Segments is type of segments.
 * \param segments is an instance of segments.
 * \param str is output stream.
 * \return reference to the output stream.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsPrintingExample-1.cpp
 * \par Output
 * \include SegmentsPrintingExample-1.out
 */
template< typename Segments >
std::ostream& printSegments( const Segments& segments, std::ostream& str )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;

   auto segmentsCount = segments.getSegmentsCount();
   str << " [";
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ )
   {
      auto segmentSize = segments.getSegmentSize( segmentIdx );
      str << " " << segmentSize;
      if( segmentIdx < segmentsCount )
         str << ",";
   }
   str << " ] ";
   return str;
}


/**
 * \brief Print segments with related content.
 *
 * \tparam Segments is type of segments.
 * \tparam Fetch is a lambda function for reading related data.
 * \param segments is an instance of segments.
 * \param fetch is an instance of lambda function reading related data. It is supposed to defined as
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) -> ValueType { return data_view[ globalIdx ]; };
 * ```
 *
 * \param str is output stream.
 * \return reference to the output stream.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsPrintingExample-2.cpp
 * \par Output
 * \include SegmentsPrintingExample-2.out
 */
template< typename Segments,
          typename Fetch >
std::ostream& printSegments( const Segments& segments, Fetch&& fetch, std::ostream& str )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   using ValueType = decltype( fetch( IndexType() ) );

   TNL::Containers::Array< ValueType, DeviceType, IndexType > aux( 1 );
   auto view = segments.getConstView();
   for( IndexType segmentIdx = 0; segmentIdx < segments.getSegmentsCount(); segmentIdx++ )
   {
      str << "Seg. " << segmentIdx << ": [ ";
      auto segmentSize = segments.getSegmentSize( segmentIdx );
      for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
      {
         aux.forAllElements( [=] __cuda_callable__ ( IndexType elementIdx, double& v ) mutable {
            v = fetch( view.getGlobalIndex( segmentIdx, localIdx ) );
         } );
         auto value = aux.getElement( 0 );
         str << value;
         if( localIdx < segmentSize - 1 )
            str << ", ";
      }
      str << " ] " << std::endl;
   }
   return str;
}


template< typename Segments,
          typename Fetch >
struct SegmentsPrinter
{
   SegmentsPrinter( const Segments& segments, Fetch& fetch )
   : segments( segments ), fetch( fetch ) {}

   std::ostream& print( std::ostream& str ) const { return printSegments( segments, fetch, str ); }

   protected:

   const Segments& segments;

   Fetch& fetch;
};

template< typename Segments,
          typename Fetch >
std::ostream& operator<<( std::ostream& str, const SegmentsPrinter< Segments, Fetch >& printer )
{
   return printer.print( str );
}

      } // namespace Segments
   } // namespace Algorithms
} // namespace TNL
