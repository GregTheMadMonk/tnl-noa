INSTALL( FILES tnl-sparse-matrix-benchmark.cfg.desc
               tnl-matrix-solvers-benchmark.cfg.desc
               tnl-benchmark-spmv.cfg.desc
         DESTINATION share/tnl-${tnlVersion} )
INSTALL( FILES matrix-market
               florida-matrix-market
               get-matrices
               convert-matrices
               draw-matrices
               run-matrix-solvers-benchmark
               run-tnl-benchmark-spmv
               cuda-profiler.conf
               process-cuda-profile.pl 
               DESTINATION share/tnl-${tnlVersion}/benchmark-scripts )

INSTALL( FILES tnl-run-spmv-benchmark
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
                                                                   