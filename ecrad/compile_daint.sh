#!/bin/bash -l

. ../dev/daint_env.sh || exit

INFERO_ROOT="/project/d121/ruestefa/spack/daint/infero-0.1.2/gcc-9.3.0/5luvhyaplpwvxbn3tbxb3tpfyti4kl2l"
LIBS='-linferoapi -linferof -lnetcdff -lnetcdf -lmpichf90'
FCFLAGS="-cpp -Wall -pedantic -fbacktrace -ffree-line-length-256 -I${NETCDF_DIR}/include -I${INFERO_ROOT}/module -Wl,-rpath,${INFERO_ROOT}/lib64 -L${NETCDF_DIR}/lib -I${MPICH_DIR}/include -I${INFERO_ROOT}/include"
# FCFLAGS+=" -O0 -g"
FCFLAGS+=" -O2 -g"
LDFLAGS="-L${INFERO_ROOT}/lib64 -L${NETCDF_DIR}/lib -L${MPICH_DIR}/lib"

ftn -o multiple_fields_model.exe $FCFLAGS $LDFLAGS $LIBS ecrad_ml.f90 || exit
ftn -o performance.exe $FCFLAGS $LDFLAGS $LIBS performance.f90 || exit
