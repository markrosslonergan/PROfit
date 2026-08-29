add_executable(PROfit
    PROfit.cxx
    PROfit_cli.cxx
    PROfit_setup.cxx
    PROfit_process.cxx
    PROfit_plot.cxx
    PROfit_fit.cxx
    PROfit_global.cxx
    PROfit_profile.cxx
    PROfit_surface.cxx
    PROfit_fc.cxx
    PROfit_afc.cxx
    PROfit_mcmc.cxx
    PROfit_bench.cxx
    PROfit_protest.cxx
    PROfit_proletariat.cxx
    ${CMAKE_BINARY_DIR}/PROfit_dict.cxx )
target_include_directories(PROfit PRIVATE ${SBNANAOBJ_INCLUDE_DIR} ${HIGHFIVE_INCLUDE_DIR})
target_compile_features(PROfit PRIVATE cxx_std_17)
target_link_directories(PROfit_dict PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(PROfit PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp PROfit_dict ${libraries})

if(MPI_ENABLE)
    add_executable(PROmfa4u PROmfa4u.cxx)
    target_compile_features(PROmfa4u PRIVATE cxx_std_17)
    target_link_libraries(PROmfa4u PRIVATE PROfitLib tinyxml2 eigen Boost::serialization lbfgspp )
endif(MPI_ENABLE)

install(TARGETS PROfit 
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR})
