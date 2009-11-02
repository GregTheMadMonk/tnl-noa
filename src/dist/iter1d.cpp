/***************************************************************************
                          iter1d.cpp  -  description
                             -------------------
    begin                : 2005/08/09
    copyright            : (C) 2005 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <mdiff.h>
#include "debug.h"
#include "iter1d.h"
#include "mDistAux.h"

/*void mDstIter1D( const double& t,
                 mGrid1D** _phi,
                 mGrid1D** _f_phi,
                 void* _aux,
                 const mESolverInfo1D& info )
{
   DBG_FUNCTION_NAME( "", "f_1D" );
         
   assert( _aux );
   
   mGrid1D& phi = * _phi[ 0 ];
   mGrid1D& f_phi = * _f_phi[ 0 ];

   mDistAux< mGrid1D >* aux = ( mDistAux< mGrid1D >* ) _aux;
   mDistIterMethod method = aux -> method;
   const double& epsilon = aux -> epsilon;

   int size = phi. Size();
   double h = phi. H();
   int i;

   mGrid1D& mod_grad = aux -> mod_grad;
   mGrid1D& laplace = aux -> laplace;

   if( method == mDstRegularisedScheme )
   {
      for( i = 0; i < size; i ++ )
         laplace( i ) = phi. Laplace( i );
      for( i = 0; i < size; i ++ )
         mod_grad( i ) = 
            0.5 * ( fabs( phi. Partial_x_f( i ) ) +
                    fabs( phi. Partial_x_b( i ) ) );
      ExtrapolateEdges( &laplace );
      ExtrapolateEdges( &mod_grad );
      for( i = 0; i < size; i ++ )
         f_phi( i ) = epsilon * h * laplace( i ) + 
            sign( phi( i ) ) * ( 1.0 - mod_grad( i ) );
      return;
   }
 
   for( i = 0; i < size; i ++ )
   {
      double grxf = phi. Partial_x_f( i );
      double grxb = phi. Partial_x_b( i );
      double grxfp = Max( grxf, 0.0 );
      double grxfn = Min( grxf, 0.0 );
      double grxbp = Max( grxb, 0.0 );
      double grxbn = Min( grxb, 0.0 );
      double d_plus, d_minus;
      switch( method )
      {
         case mDstUpwindScheme:
            d_plus = sqrt( grxbp * grxbp + grxfn * grxfn );
            d_minus = sqrt( grxfp * grxfp + grxbn * grxbn );
         break;
         case mDstGodunovScheme:
            d_plus = Max( grxbp, -1.0 * grxfn );
            d_minus = Max( grxfp, -1.0 * grxbn );
         break;
         default:
         break;
      }
      double F_plus = Max( sign( phi( i ) ), 0.0 );       
      double F_minus = Min( sign( phi( i ) ), 0.0 );
      f_phi( i ) = sign( phi( i ) ) - ( F_plus * d_plus + F_minus * d_minus );
   }
}
//--------------------------------------------------------------------------
void IterDist1D( mGrid1D* phi, mConfig& config )
{
   DBG_FUNCTION_NAME( "", "IterDist1D" );

   mESolver1D solver;
   
   double init_t = config. Get< double >( "init-t" );
   double tau = config. Get< double >( "init-tau" );
   double final_t = config. Get< double >( "final-t" );
   double epsilon = config. Get< double >( "epsilon" );
   double output_period = config. Get< double >( "output-period" );
   double adaptivity = config. Get< double >( "adaptivity" );
   int step = config. Get< int >( "init-step" );
   const m_char* method_name = 
      config. Get< mString >( "method-name" ). Data();
   const m_char* output_file = 
      config. Get< mString >( "output-file" ). Data();
   const m_char* output_file_ending = 
      config. Get< mString >( "output-file-ending" ). Data();
   
   double t( init_t );
   double break_t = t + output_period;
   if( ! output_period ) output_period = final_t - init_t;
   if( ! tau ) tau = output_period;
   if( ! adaptivity ) adaptivity = 1.0e-4;

   mDistIterMethod method( mDstNone );
   if( strcmp( method_name, "iter-regularised" ) == 0 )
      method = mDstRegularisedScheme;
   if( strcmp( method_name, "iter-upwind" ) == 0 )
      method = mDstUpwindScheme;
   if( strcmp( method_name, "iter-godunov" ) == 0 )
      method = mDstGodunovScheme;
   if( ! method )
   {
      cerr << "Uknown method name '" << method_name << "' !" << endl;
      return;
   }
   
   int size = phi -> Size();
   cout << "problem name: " << config. Get< mString >( "problem-name" ) << endl;
   cout << "Dim = 1 " << endl;
   cout << "Size = " << size << endl;
   cout << "h = " << phi -> H() << endl;
   cout << "init_t = " << init_t << " final_t = " << final_t << endl;
   cout << "Method name: " << method_name << endl;
   cout << "Adaptivity: " << adaptivity << endl;
   if( method == mDstRegularisedScheme )
      cout << "epsilon = " << epsilon << endl;
   cout << "Initial step " << step << endl;
   cout << "output file : " << output_file << endl
        << "output file ending : " << output_file_ending << endl;
   cout << "Starting computation" << endl;
 
   mDistAux< mGrid1D > aux( *phi, method, epsilon );

   solver. Init( "merson",
                 *phi,
                 0, //SaveFunction,
                 0, //&saver, 
                 0 );
   solver. SetT( t );
   //solver. SetFinalT( final_t );
   solver. SetTau( tau );
   solver. SetAdaptivity( adaptivity );
   
   m_char file_name[ 1024 ];
   if( output_file )
   {
      FileNumberEnding( file_name,
                        output_file,
                        step,
                        5,
                        output_file_ending );
      cout << endl << "Writing file: " << file_name << endl;
      assert( phi );
      phi -> DrawFunction( file_name );
   }
   solver. StartTiming();
   while( t < final_t )
   {
      solver. SetBreakT( break_t );
      solver. Solve( &phi, mDstIter1D, 0, &aux );
      t = solver. GetT();
      break_t += output_period;
      step ++;
      if( output_file )
      {
         FileNumberEnding( file_name,
                           output_file,
                           step,
                           5,
                           output_file_ending );
         cout << endl << "Writing file: " << file_name << endl;
         assert( phi );
         phi -> DrawFunction( file_name );
      }
      
      if( output_curve_file )
      {
         FileNumberEnding( file_name,
                           output_curve_file,
                           step,
                           6,
                           file_ending );
         cout << endl << "Writing file: " << file_name << endl;
         curve. Identify( *_phi, curve_level_set );
         curve. Draw( file_name );
         curve. Delete();
      }

   }
}*/
