/***************************************************************************
                          tnlMatrixWriter_impl.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Matrix >
bool tnlMatrixWriter< Matrix >::writeToGnuplot( std::ostream str,
                                                const Matrix& matrix,
                                                bool verbose )
{
   for( IndexType row = 0; row < matrix.getRows(); row ++ )
   {
      for( IndexType column = 0; column < matrix.getColumns(); column ++ )
      {
         RealType elementValue = maytrix.getElement( row, column );
         if(  elementValue != ( RealType ) 0.0 )
            str << column << " " << getSize() - row << " " << elementValue << endl;
      }
      if( verbose )
         cout << "Drawing the row " << row << "      \r" << flush;
   }
   if( verbose )
      cout << endl;
   return true;
}

template< typename Matrix >
bool tnlMatrixWriter< Matrix >::writeToEps( std::ostream str,
                                            const Matrix& matrix,
                                            bool verbose )
{
   const int elementSize = 10;
   if( ! writeEpsHeader( str, matrix, elementSize ) )
      return false;
   if( !writeEpsBody( str, matrix, elementSize, verbose ) )
      return false;

   str << "showpage" << endl;
   str << "%%EOF" << endl;

   if( verbose )
      cout << endl;
   return true;
}

template< typename Matrix >
bool tnlMatrixWriter< Matrix >::writeEpsHeader( std::ostream str,
                                                const Marix& matrix,
                                                const int elementSize )
{
   const double scale = elementSize * Max( matrix.getRows(), matrix.getColumns() );
   str << "%!PS-Adobe-2.0 EPSF-2.0" << endl;
   str << "%%BoundingBox: 0 0 " << scale << " " << scale << endl;
   str << "%%Creator: TNL" << endl;
   str << "%%LanguageLevel: 2" << endl;
   str << "%%EndComments" << endl << endl;
   str << "0 " << scale << " translate" << endl;
   return true;
}

template< typename Matrix >
bool tnlMatrixWriter< Matrix >::writeEpsBody( std::ostream str,
                                              const Marix& matrix,
                                              const int elementSize )
{
   IndexType lastRow( 0 ), lastColumn( 0 );
   for( IndexType row = 0; row < getSize(); row ++ )
   {
      for( IndexType column = 0; column < getSize(); column ++ )
      {
         RealType elementValue = getElement( row, column );
         if( elementValue != ( RealType ) 0.0 )
         {
            str << ( column - lastColumn ) * elementSize
                << " " << -( row - lastRow ) * elementSize
                << " translate newpath 0 0 " << elementSize << " " << elementSize << " rectstroke" << endl;
            lastColumn = column;
            lastRow = row;
         }
      }
      if( verbose )
         cout << "Drawing the row " << row << "      \r" << flush;
   }
   return true;
}

} // namespace TNL
