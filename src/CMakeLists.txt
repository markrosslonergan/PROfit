file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/inc/*.h")

if(MPI_ENABLE)
    add_library(PROfitLib PROconfig.cxx PROspec.cxx PROdata.cxx PROmfa.cxx PROtocall.cxx PROcreate.cxx PROsyst.cxx PROcess.cxx PROchi.cxx PROCNP.cxx PROpoisson.cxx PROsurf.cxx PROfc.cxx PROfitter.cxx PROplot.cxx PROunblind.cxx MurmurHash3 ${CMAKE_BINARY_DIR}/PROfit_dict.cxx ${HEADER_LIST})
    target_include_directories(PROfitLib PUBLIC ${PROJECT_SOURCE_DIR}/inc ${install_dir}/src/mfa/include/ ${MPI_INCLUDE_PATH} ${DIY_INCLUDE_DIR} ${FMT_INCLUDE_DIR} ${SBNANAOBJ_INCLUDE_DIR})
    target_link_libraries(PROfitLib PRIVATE Boost::serialization eigen tinyxml2 lbfgspp PROfit_dict ${ROOT_LIBRARIES} ${libraries})
    target_compile_features(PROfitLib PUBLIC cxx_std_17)
else ()
    add_library(PROfitLib PROconfig.cxx PROspec.cxx PROdata.cxx PROtocall.cxx PROcreate.cxx PROsyst.cxx PROcess.cxx PROchi.cxx PROCNP.cxx PROpoisson.cxx PROsurf.cxx PROfc.cxx PROfitter.cxx PROplot.cxx PROunblind.cxx MurmurHash3.cpp ${CMAKE_BINARY_DIR}/PROfit_dict.cxx ${HEADER_LIST})
    target_include_directories(PROfitLib PUBLIC ${PROJECT_SOURCE_DIR}/inc  ${SBNANAOBJ_INCLUDE_DIR} ${HDF5_INCLUDE_DIR})
    target_link_libraries(PROfitLib PRIVATE eigen Boost::serialization tinyxml2 lbfgspp PROfit_dict ${ROOT_LIBRARIES} ${libraries})
    target_compile_features(PROfitLib PUBLIC cxx_std_17)
endif(MPI_ENABLE) 

# Install PROfit
install(TARGETS PROfitLib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# IDEs should put the headers in a nice place
source_group(
    TREE "${PROJECT_SOURCE_DIR}/inc"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})


