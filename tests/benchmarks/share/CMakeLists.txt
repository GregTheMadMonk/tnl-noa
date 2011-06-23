INSTALL( FILES tnl-sparse-matrix-benchmark.cfg.desc
         DESTINATION share/tnl-${tnlVersion} )
INSTALL( FILES matrix-market
               florida-matrix-market
               get-matrices
               convert-matrices
               draw-matrices
               run-sparse-matrix-benchmark
               run-matrix-solvers-benchmark
               cuda-profiler.conf
               process-cuda-profile.pl 
               DESTINATION share/tnl-${tnlVersion}/benchmark-scripts )


                                                                   