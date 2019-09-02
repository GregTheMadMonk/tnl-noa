double sequentialSum( const double* a, const int size )
{
   auto fetch = [=] (int i)->double { return a[ i ]; };
   auto reduce = [] (double& x, const double& y) { x += y; };

   double sum( 0.0 );
   for( int i = 0; i < size; i++ )
       reduce( sum, fetch( i ) );
   return sum;
}

