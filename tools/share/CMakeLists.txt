file(COPY mdiff-err-norms.cfg.desc DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )
file(COPY tnl-grid-view.cfg.desc DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )
file(COPY tnlcurve2gnuplot.cfg.desc DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )
file(COPY tnl-matrix-convert.cfg.desc DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )

INSTALL( FILES mdiff-err-norms.cfg.desc
               tnl-grid-view.cfg.desc
               tnlcurve2gnuplot.cfg.desc
               tnl-matrix-convert.cfg.desc
         DESTINATION share/tnl-${tnlVersion} )