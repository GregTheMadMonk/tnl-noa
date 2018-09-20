open( INPUT, "$ARGV[0]" )
    or die "Can not open file $ARGV[ 0 ]";
$blockSize = 0;
$testNumber = 0;
while( $line = <INPUT> )
{
	if( $line =~ m/.*sparseCSRMatrixVectorProductKernel.*threadblocksize=\[ (.*), 1, 1 \] occupancy=\[ (.*) \] tex_cache_hit=\[ (.*) \] tex_cache_miss=\[ (.*) \] gld_incoherent=\[ (.*) \] gst_incoherent=\[ (.*) \].*/ )
	{
		if( $blockSize != $1 )
		{
           $blockSize = $1;
 	   	   $occupancy{$testNumber} = $2;
 	   	   $texCacheHit{$testNumber} = $3;
 	   	   $texCacheMiss{$testNumber} = $4;
 	   	   $gldIncoherent{$testNumber} = $5;
 	   	   $gstIncoherent{$testNumber} = $6;
	   	   $testNumber = $testNumber + 1;
	   }
	}
}
close( INPUT );

print "There were $testNumber tests.";

open( LOG, ">>$ARGV[1]" )
    or die "Can not open file $ARGV[1]";
printf LOG "| %97s |", $ARGV[ 0 ];
$testOutput = 0;
while( $testOutput < $testNumber )
{
	printf LOG "%10.3f |", $occupancy{$testOutput};
	printf LOG "%10.3f |", $texCahceHit{$testOutput};
	printf LOG "%10.3f |", $texCacheMiss{$testOutput};
	printf LOG "%10.3f |", $gldIncoherent{$testOutput};
	printf LOG "%10.3f |", $gstIncoherent{$testOutput};
	$testOutput = $testOutput + 1; 
}
print LOG "\n";
close( LOG );    
    
    
	
