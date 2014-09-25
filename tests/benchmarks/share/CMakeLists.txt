INSTALL( FILES matrix-market
               florida-matrix-market
               get-matrices
               convert-matrices
               draw-matrices
               run-matrix-solvers-benchmark
               run-tnl-benchmark-spmv
               run-tnl-benchmark-linear-solvers
               cuda-profiler.conf
               process-cuda-profile.pl 
               tnl-log-to-html.py
               DESTINATION share/tnl-${tnlVersion}/benchmark-scripts )

INSTALL( FILES tnl-run-spmv-benchmark
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
                                                                   