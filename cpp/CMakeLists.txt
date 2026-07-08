cmake_minimum_required(VERSION 3.20)

project(quantfinlab_kernels LANGUAGES NONE)

option(quantfinlab_build_cpp "Build the optional QuantFinLab C++ pricing kernels" ON)

if(DEFINED QUANTFINLAB_BUILD_CPP)
    set(quantfinlab_build_cpp "${QUANTFINLAB_BUILD_CPP}" CACHE BOOL "Build the optional QuantFinLab C++ pricing kernels" FORCE)
endif()

if(quantfinlab_build_cpp)
    enable_language(CXX)

    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(OpenMP QUIET)

    pybind11_add_module(_kernels
        bindings/kernels_bindings.cpp
        src/american_tree.cpp
        src/american_pde.cpp
        src/american_lsm.cpp
        src/fourier.cpp
    )

    target_include_directories(_kernels PRIVATE include ${Python_NumPy_INCLUDE_DIRS})
    target_compile_features(_kernels PRIVATE cxx_std_17)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(MSVC)
        target_compile_options(_kernels PRIVATE /O2 /EHsc)
    else()
        target_compile_options(_kernels PRIVATE -O3)
    endif()

    install(TARGETS _kernels LIBRARY DESTINATION quantfinlab)
endif()
