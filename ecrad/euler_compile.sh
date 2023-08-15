#!/bin/bash -l

. ../dev/euler_env.sh

INFERO_ROOT='../../infero_build/builds/infero'
#LIBS='-linferoapi -linferof -lnetcdff -lnetcdf'
LIBS='-linferoapi -linferof -lnetcdff'
FCFLAGS="-cpp -Wall -pedantic -fbacktrace -O2 -g -ffree-line-length-256 -I${NETCDF_FORTRAN_ROOT}/include -I${INFERO_ROOT}/module -Wl,-rpath,${INFERO_ROOT}/lib -L${NETCDF_FORTRAN_ROOT}/lib"
LDFLAGS="-L${INFERO_ROOT}/lib -L${NETCDF_FORTRAN_ROOT}/lib"

gfortran -o multiple_fields_model.exe $FCFLAGS $LDFLAGS $LIBS ecrad_ml.f90 
