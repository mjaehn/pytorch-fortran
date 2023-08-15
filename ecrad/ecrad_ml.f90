program ecrad_ml
  use inferof
  use iso_c_binding, only : c_float, c_null_char
  implicit none

  ! Command line arguments
  character(1024) :: model_path
  character(1024) :: model_type
  character(1024) :: yaml_config

  ! model of infero model
  type(infero_model) :: model
  type(infero_tensor_set) :: iset
  type(infero_tensor_set) :: oset

  ! parameters
  REAL (c_float), PARAMETER ::  pi = 3.14159265358979323846264338327950288
  REAL (c_float), PARAMETER ::  rad2deg   = 180.0/pi

  ! no. grid points passed to the model
  ! set to zero to use all available grid points
  integer :: batch_size = 100
  !integer :: batch_size = 81919
  !integer :: batch_size = 81920
  !integer :: batch_size = 0
  integer :: s_idx, e_idx

  ! no. vertical levels passed to the model
  ! set to zero to use all available levels
  ! positive/negative values select levels from the TOA/surface
  integer :: n_lev = 70
  integer :: s_lev, e_lev

  ! character(100), parameter :: input_tensor_name_2d = "serving_default_input_22"
  ! character(100), parameter :: input_tensor_name_3d = "serving_default_input_23"
  character(100), parameter :: input_tensor_name_2d = "serving_default_input_2d"
  character(100), parameter :: input_tensor_name_3d = "serving_default_input_3d"
  character(100), parameter :: output_tensor_name = "StatefulPartitionedCall"

  integer, parameter :: nsteps = 4
  integer, parameter :: nflxs = 4
  integer, parameter :: nvars_2d_rd = 8
  integer, parameter :: nvars_2d_in = 8
  integer, parameter :: nvars_3d_rd = 10
  integer, parameter :: nvars_3d_in = 6
  integer, parameter :: dummy_dim = 1

  ! input dimensions
  integer, parameter :: ndims = 6
  integer, parameter :: idim_time = 1
  integer, parameter :: idim_ncells = 2
  integer, parameter :: idim_vertices = 3
  integer, parameter :: idim_height = 4
  integer, parameter :: idim_bnds = 5
  integer, parameter :: idim_height_2 = 6

  ! input and output tensors
  real(c_float), allocatable :: input_3d(:,:,:,:)
  real(c_float), allocatable :: input_2d(:,:,:)
  real(c_float), allocatable :: pred_flx(:,:,:)

  ! netcdf
  character(1024) :: netcdf_data_file, icon_grid
  character(50) :: dim_name(ndims), grid_dim_name(14)
  character(19) :: timestamp(nsteps)
  integer :: dim_len(ndims), grid_dim_len(14)

  ! indices
  integer :: i, k, id_d, step
  integer :: nc_time_idx(nsteps)
  integer :: counter(nsteps)

  ! data fields
  real(c_float), allocatable :: from_netcdf_2d(:,:,:)
  real(c_float), allocatable :: from_netcdf_3d(:,:,:,:)
  !real(c_float), allocatable :: neighbor_cell_index(:,:)
  real(c_float), allocatable :: abs_diff(:,:,:,:)
  real(c_float), allocatable :: swflx(:,:,:,:)
  real(c_float), allocatable :: lwflx(:,:,:,:)
  real(c_float), allocatable :: clat(:), clon(:)
  real(c_float), allocatable :: lat(:), lon(:)
  real(c_float) :: mean_absolute_error(nsteps,nflxs)

  ! scalars
  real(c_float) :: percentage, lat_min, lat_max, lon_min, lon_max

  ! Get CL arguments
  CALL get_command_argument(1, model_path)
  CALL get_command_argument(2, model_type)

  ! DEFINE PERIOD (FULL SOLAR CYCLE)
  ! correspond to file nr. 5
  timestamp(1) = "2000-01-29 00:00:00"
  timestamp(2) = "2000-01-29 06:00:00"
  timestamp(3) = "2000-01-29 12:00:00"
  timestamp(4) = "2000-01-29 18:00:00"

  ! store NetCDF indices of timestamps defined above
  nc_time_idx(1) = 1
  nc_time_idx(2) = 2
  nc_time_idx(3) = 3
  nc_time_idx(4) = 4

  ! READ-IN ICON GRID INFORMATION AND COMPUTE DOMAIN EXTENT
  icon_grid='icon_grid.nc'

  ! read icon grid file dimensions
  write(*,'(a)') 'read icon grid dimensions file ' // trim(icon_grid)
  call get_nc_dims(icon_grid,grid_dim_name,grid_dim_len,14)

  ! fields with icon-grid information
  write(*,'(a)') 'allocate fields with icon-grid information'
  !ALLOCATE(neighbor_cell_index(grid_dim_len(1), grid_dim_len(5)))
  ALLOCATE(clat(grid_dim_len(1)))
  ALLOCATE(clon(grid_dim_len(1)))
  ALLOCATE(lat(batch_size))
  ALLOCATE(lon(batch_size))

  !call read_nc_2d(icon_grid, "neighbor_cell_index", neighbor_cell_index(:,:), grid_dim_len(1), grid_dim_len(5))
  call read_nc_1d(icon_grid, "clat", clat, grid_dim_len(1))
  call read_nc_1d(icon_grid, "clon", clon, grid_dim_len(1))

  s_idx = 1
  e_idx = batch_size

  ! compute domain extent
  lon(1:batch_size) = rad2deg * clon(s_idx:e_idx)
  lat(1:batch_size) = rad2deg * clat(s_idx:e_idx)

  lon_max = MAXVAL(lon(:))
  lon_min = MINVAL(lon(:))
  lat_max = MAXVAL(lat(:))
  lat_min = MINVAL(lat(:))

  ! READ INPUT DATA FOR MODEL
  netcdf_data_file='input_data.nc'

  ! read data dimensions
  write(*,'(a)') 'read data dimensions from ' // trim(netcdf_data_file)
  call get_nc_dims(netcdf_data_file, dim_name, dim_len, ndims)

  write(*,'(A)') ''
  do i=1,ndims
    write(*,'(a,i1,a,i6)') 'dim_len(', i ,') =', dim_len(i)
  end do
  write(*,'(A)') ''
  write(*,'(a,i6)') 'dim_len(idim_time    ) = ', dim_len(idim_time)
  write(*,'(a,i6)') 'dim_len(idim_ncells  ) = ', dim_len(idim_ncells)
  write(*,'(a,i6)') 'dim_len(idim_vertices) = ', dim_len(idim_vertices)
  write(*,'(a,i6)') 'dim_len(idim_height  ) = ', dim_len(idim_height)
  write(*,'(a,i6)') 'dim_len(idim_bnds    ) = ', dim_len(idim_bnds)
  write(*,'(a,i6)') 'dim_len(idim_height_2) = ', dim_len(idim_height_2)
  write(*,'(A)') ''

  ! SR/TODO properly deal w/ height_2 fields (now using idim_height everywhere)
  if (dim_len(idim_height) /= dim_len(idim_height_2)) then
    write(*,'(a,2i6)') 'ERROR: height != height_2: ', dim_len(idim_height), dim_len(idim_height_2)
    call ABORT()
  endif

  ! set/check batch_size, s_idx and e_idx
  if (batch_size == 0) then
    batch_size = dim_len(idim_ncells)
  elseif (batch_size < 0) then
    write(*,'(a)') 'WARNING: batch_size should be positive'
    batch_size = -batch_size
  endif
  if (batch_size > dim_len(idim_ncells)) then
    write(*,'(a)') 'WARNING: batch_size too large'
    batch_size = dim_len(idim_ncells)
  endif
  s_idx = 1  ! SR/TMP
  e_idx = batch_size  ! SR/TMP
  write(*,'(a,i6)') 'batch_size : ', batch_size
  write(*,'(a,i6)') 's_idx      : ', s_idx
  write(*,'(a,i6)') 'e_idx      : ', e_idx

  ! set/check n_lev, s_lev and e_lev
  if (n_lev == 0) then
    n_lev = dim_len(idim_height)
    s_lev = 1
    e_lev = n_lev
  else
    if (ABS(n_lev) > dim_len(idim_height)) then
      write(*,'(a)') 'WARNING: n_lev too large'
      n_lev = SIGN(dim_len(idim_height), n_lev)
    endif
    if (n_lev > 0) then
      s_lev = 1
      e_lev = n_lev
    else
      n_lev = -n_lev
      e_lev = dim_len(idim_height)
      s_lev = e_lev - n_lev
    endif
  endif
  write(*,'(a,i6)') 'n_lev      : ', n_lev
  write(*,'(a,i6)') 's_lev      : ', s_lev
  write(*,'(a,i6)') 'e_lev      : ', e_lev

  write(*,'(a)') ''

  write(*,'(a)') 'allocate fields for NetCDF data'
  ALLOCATE(from_netcdf_2d(dim_len(idim_ncells), dim_len(idim_time), nvars_2d_rd))
  ALLOCATE(from_netcdf_3d(dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time), nvars_3d_rd))
  from_netcdf_2d(:,:,:)   = 0.0_c_float
  from_netcdf_3d(:,:,:,:) = 0.0_c_float

  write(*,'(a)') 'allocate fields for model data'
  ALLOCATE(input_2d(batch_size, dummy_dim, nvars_2d_in))
  ALLOCATE(input_3d(batch_size, n_lev, dummy_dim, nvars_3d_in))
  ALLOCATE(pred_flx(batch_size, n_lev, 4))
  ALLOCATE(swflx(batch_size, n_lev, nsteps, 2))
  ALLOCATE(lwflx(batch_size, n_lev, nsteps, 2))
  input_2d(:,:,:)   = 0.0_c_float
  input_3d(:,:,:,:) = 0.0_c_float
  pred_flx(:,:,:)   = 0.0_c_float
  swflx   (:,:,:,:) = 0.0_c_float
  lwflx   (:,:,:,:) = 0.0_c_float

  ! 2d fields
  write(*,'(a)') 'read 2D fields'
  call read_nc_2d(netcdf_data_file, "pres_sfc",   from_netcdf_2d(:,:,1), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "cosmu0",     from_netcdf_2d(:,:,2), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "qv_s",       from_netcdf_2d(:,:,3), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "albvisdir",  from_netcdf_2d(:,:,4), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "albnirdir",  from_netcdf_2d(:,:,5), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "tsfctrad",   from_netcdf_2d(:,:,6), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "albvisdif",  from_netcdf_2d(:,:,7), dim_len(idim_ncells), dim_len(idim_time))
  call read_nc_2d(netcdf_data_file, "albnirdif",  from_netcdf_2d(:,:,8), dim_len(idim_ncells), dim_len(idim_time))

  ! 3d fields
  write(*,'(a)') 'read 3D fields'
  call read_nc_3d(netcdf_data_file, "clc",        from_netcdf_3d(:,:,:, 1), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "temp",       from_netcdf_3d(:,:,:, 2), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "pres",       from_netcdf_3d(:,:,:, 3), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "qc",         from_netcdf_3d(:,:,:, 4), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "qi",         from_netcdf_3d(:,:,:, 5), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "qv",         from_netcdf_3d(:,:,:, 6), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "lwflx_up",   from_netcdf_3d(:,:,:, 7), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "lwflx_dn",   from_netcdf_3d(:,:,:, 8), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "swflx_up",   from_netcdf_3d(:,:,:, 9), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))
  call read_nc_3d(netcdf_data_file, "swflx_dn",   from_netcdf_3d(:,:,:,10), dim_len(idim_ncells), dim_len(idim_height), dim_len(idim_time))

  ! SET UP INFERO AND ASSIGN IN/OUT TENSORS

  ! init infero
  write(*,'(a)') 'initialize infero'
  call infero_check(infero_initialise())

  ! YAML config string -> LW/SW
  yaml_config = "---"//NEW_LINE('A') &
    & //"  path: "//TRIM(model_path)//NEW_LINE('A') &
    & //"  type: "//TRIM(model_type)//c_null_char

  ! get a infero model for LW/SW
  call infero_check(model%initialise_from_yaml_string(yaml_config))

  ! init tensor sets
  call infero_check(iset%initialise())
  call infero_check(oset%initialise())
  call infero_check(iset%push_tensor(input_3d, trim(input_tensor_name_3d)))
  call infero_check(iset%push_tensor(input_2d, trim(input_tensor_name_2d)))
  call infero_check(oset%push_tensor(pred_flx, trim(output_tensor_name)))

  ! TIMESTEP
  write(*,'(a)') 'run model'
  DO step=1,nsteps
    write(*,'(a,i1)') 'STEP ', step

    ! update input-tensors
    input_2d(1:batch_size,          dummy_dim, 1:nvars_2d_in) = from_netcdf_2d(s_idx:e_idx,              nc_time_idx(step), 1:nvars_2d_in)
    input_3d(1:batch_size, 1:n_lev, dummy_dim, 1:nvars_3d_in) = from_netcdf_3d(s_idx:e_idx, s_lev:e_lev, nc_time_idx(step), 1:nvars_3d_in)

    ! apply model
    write(*,'(a)') 'apply model'
    call infero_check(model%infer(iset, oset))

    !CALL infero_check(iset%print())
    !CALL infero_check(oset%print())

    ! store results in permanent fields
    write(*,'(a)') 'store results in permanent fields'
    lwflx(1:batch_size, 1:n_lev, step, 1:2) = pred_flx(1:batch_size, 1:n_lev, 1:2)
    swflx(1:batch_size, 1:n_lev, step, 1:2) = pred_flx(1:batch_size, 1:n_lev, 3:4)

    ! input
    write(*,'(a)') 'compute stats of input vars'
    CALL stats_2d("input_clc",  input_3d(:,:,1,1))
    CALL stats_2d("input_temp", input_3d(:,:,1,2))
    CALL stats_2d("input_pres", input_3d(:,:,1,3))
    CALL stats_2d("input_qc",   input_3d(:,:,1,4))
    CALL stats_2d("input_qi",   input_3d(:,:,1,5))
    CALL stats_2d("input_qv",   input_3d(:,:,1,6))

    ! output
    write(*,'(a)') 'compute stats of output vars'
    CALL stats_2d("lwflx_up", pred_flx(:,:,1))
    CALL stats_2d("lwflx_dn", pred_flx(:,:,2))
    CALL stats_2d("swflx_up", pred_flx(:,:,3))
    CALL stats_2d("swflx_dn", pred_flx(:,:,4))
  ENDDO

  ! DOMAIN EXTENT
  write(*,'(A,I7,A)') 'Domain extent for batch_size', batch_size,':'
  write(*,'(A,F8.2,A,F8.2)') '  latmax:', lat_max, ' latmin', lat_min
  write(*,'(A,F8.2,A,F8.2)') '  lonmax:', lon_max, ' lonmin', lon_min

  ! STATISTICS
  write(*,'(A)') ''
  write(*,'(A)') 'STATISTICS'
  write(*,'(A)') ''

  ! count negative SW fluxes
  counter(:) = 0
  DO step=1,nsteps
    DO i=1,batch_size
      DO k=1,n_lev
        DO id_d=1,2
          IF(swflx(i,k,id_d,step) < 0.0) THEN
            counter(step) = counter(step) + 1
          ENDIF
        ENDDO
      ENDDO
    ENDDO
  ENDDO

  write(*,'(I7,A)') batch_size*n_lev*2,' datapoints scanned for each step of SW-flux'
  DO step=1,nsteps
    percentage = MAX(0.0,100.0 * REAL(counter(step))/REAL((batch_size*n_lev*2)))
    write(*,'(F6.1,A,A)') percentage, '% values below 0.0 for ',timestamp(step)
  ENDDO

  ! absolute difference
  ALLOCATE(abs_diff(batch_size, n_lev, nsteps, nflxs))

  write(*,'(A)') ''
  write(*,'(A)') '  Mean Absolute Error (MAE):'
  DO step=1,nsteps
    abs_diff(1:batch_size, 1:n_lev, step, 1) = ABS(lwflx(1:batch_size, 1:n_lev, step, 1) - from_netcdf_3d(s_idx:e_idx, s_lev:e_lev, nc_time_idx(step), 7))  ! lwflux_up
    abs_diff(1:batch_size, 1:n_lev, step, 2) = ABS(lwflx(1:batch_size, 1:n_lev, step, 2) - from_netcdf_3d(s_idx:e_idx, s_lev:e_lev, nc_time_idx(step), 8))  ! lwflux_dn
    abs_diff(1:batch_size, 1:n_lev, step, 3) = ABS(swflx(1:batch_size, 1:n_lev, step, 1) - from_netcdf_3d(s_idx:e_idx, s_lev:e_lev, nc_time_idx(step), 9))  ! swflux_up
    abs_diff(1:batch_size, 1:n_lev, step, 4) = ABS(swflx(1:batch_size, 1:n_lev, step, 2) - from_netcdf_3d(s_idx:e_idx, s_lev:e_lev, nc_time_idx(step), 10)) ! swflux_dn

    ! mean values
    CALL mean_2d(abs_diff(1:batch_size, 1:n_lev, step, 1), mean_absolute_error(step, 1), batch_size, n_lev)
    CALL mean_2d(abs_diff(1:batch_size, 1:n_lev, step, 2), mean_absolute_error(step, 2), batch_size, n_lev)
    CALL mean_2d(abs_diff(1:batch_size, 1:n_lev, step, 3), mean_absolute_error(step, 3), batch_size, n_lev)
    CALL mean_2d(abs_diff(1:batch_size, 1:n_lev, step, 4), mean_absolute_error(step, 4), batch_size, n_lev)

    write(*,'(A)') ''
    write(*,'(A,A)') '    ',timestamp(step)
    write(*,'(A)') '      Upwards:'
    write(*,'(A,F8.2)') '        SW (all levels): ', mean_absolute_error(step, 3)
    write(*,'(A,F8.2)') '        LW (all levels): ', mean_absolute_error(step, 1)
    write(*,'(A)') '      Downwards:'
    write(*,'(A,F8.2)') '        SW (all levels): ', mean_absolute_error(step, 4)
    write(*,'(A,F8.2)') '        LW (all levels): ', mean_absolute_error(step, 2)
  ENDDO

  write(*,'(A)') ''
  write(*,'(A)') '  MIN/MAX values for different levels'
  DO step=1,nsteps
    write(*,'(A)') ''
    write(*,'(A,A)') '  ',timestamp(step)
    write(*,'(A)') '    Upwards:'
    write(*,'(A,F8.2,A,F8.2)') '     SW (all): MAX ', MAXVAL(swflx(1:batch_size,1:n_lev,step,1)),' MIN ', MINVAL(swflx(1:batch_size,1:n_lev,step,1))
    write(*,'(A,F8.2,A,F8.2)') '     LW (all): MAX ', MAXVAL(lwflx(1:batch_size,1:n_lev,step,1)),' MIN ', MINVAL(lwflx(1:batch_size,1:n_lev,step,1))
    write(*,'(A,F8.2,A,F8.2)') '     SW (sfc): MAX ', MAXVAL(swflx(1:batch_size,1,     step,1)),' MIN ', MINVAL(swflx(1:batch_size,1     ,step,1))
    write(*,'(A,F8.2,A,F8.2)') '     LW (sfc): MAX ', MAXVAL(lwflx(1:batch_size,1,     step,1)),' MIN ', MINVAL(lwflx(1:batch_size,1     ,step,1))
    write(*,'(A,F8.2,A,F8.2)') '     SW (toa): MAX ', MAXVAL(swflx(1:batch_size,  n_lev,step,1)),' MIN ', MINVAL(swflx(1:batch_size,  n_lev,step,1))
    write(*,'(A,F8.2,A,F8.2)') '     LW (toa): MAX ', MAXVAL(lwflx(1:batch_size,  n_lev,step,1)),' MIN ', MINVAL(lwflx(1:batch_size,  n_lev,step,1))
    write(*,'(A)') ''
    write(*,'(A)') '    Downwards:'
    write(*,'(A,F8.2,A,F8.2)') '     SW (all): MAX ', MAXVAL(swflx(1:batch_size,1:n_lev,step,2)),' MIN ', MINVAL(swflx(1:batch_size,1:n_lev,step,2))
    write(*,'(A,F8.2,A,F8.2)') '     LW (all): MAX ', MAXVAL(lwflx(1:batch_size,1:n_lev,step,2)),' MIN ', MINVAL(lwflx(1:batch_size,1:n_lev,step,2))
    write(*,'(A,F8.2,A,F8.2)') '     SW (sfc): MAX ', MAXVAL(swflx(1:batch_size,1,     step,2)),' MIN ', MINVAL(swflx(1:batch_size,1,       step,2))
    write(*,'(A,F8.2,A,F8.2)') '     LW (sfc): MAX ', MAXVAL(lwflx(1:batch_size,1,     step,2)),' MIN ', MINVAL(lwflx(1:batch_size,1,       step,2))
    write(*,'(A,F8.2,A,F8.2)') '     SW (toa): MAX ', MAXVAL(swflx(1:batch_size,  n_lev,step,2)),' MIN ', MINVAL(swflx(1:batch_size,  n_lev,step,2))
    write(*,'(A,F8.2,A,F8.2)') '     LW (toa): MAX ', MAXVAL(lwflx(1:batch_size,  n_lev,step,2)),' MIN ', MINVAL(lwflx(1:batch_size,  n_lev,step,2))
  ENDDO

  ! CLEANUP

  ! free the model
  call infero_check(model%free())

  ! finalise infero library
  call infero_check(infero_finalise())

  ! deallocate
  DEALLOCATE(swflx)
  DEALLOCATE(lwflx)
  !DEALLOCATE(neighbor_cell_index)
  DEALLOCATE(clat)
  DEALLOCATE(clon)
  DEALLOCATE(lat)
  DEALLOCATE(lon)
  DEALLOCATE(from_netcdf_3d)
  DEALLOCATE(from_netcdf_2d)
  DEALLOCATE(abs_diff)
  DEALLOCATE(input_3d)
  DEALLOCATE(input_2d)
  DEALLOCATE(pred_flx)

CONTAINS

  SUBROUTINE stats_2d(varname,input)
    use iso_c_binding, only : c_float
    REAL(c_float), INTENT(IN) :: input(:,:)
    CHARACTER(*) :: varname
    INTEGER :: nx,ny,i,j
    REAL(c_float):: mean
    nx = SIZE(input,DIM=1)
    ny = SIZE(input,DIM=2)
    mean = 0.0
    DO i=1,nx
      DO j=1,ny
        mean = mean + input(i,j)
      ENDDO
    ENDDO
    mean = mean / REAL(nx*ny)
    !JJ: uncomment for shape information
    !write(message_text,*) SHAPE(input)
    !write(message_text,'(A)') TRIM(message_text)
    !CALL message(TRIM(varname),message_text)
    write(*,'(A,A,F8.2,A,F8.2,A,F8.2)') &
      & TRIM(varname),': MEAN ',mean,' MAX ',MAXVAL(input(:,:)),' MIN ', MINVAL(input(:,:))
  END SUBROUTINE stats_2d

end program

SUBROUTINE get_nc_dims(infile,dim_name,dim_len,nr_dims)
  use netcdf
  INTEGER(KIND=4), INTENT(OUT) :: dim_len(nr_dims)
  CHARACTER(LEN=50), INTENT(OUT) :: dim_name(nr_dims)
  INTEGER(KIND=4) :: ncid,i
  CHARACTER(LEN=*), INTENT(IN) :: infile
  CALL check(nf90_open(TRIM(infile), nf90_nowrite, ncid))
  write(*,'(A,A)') 'Dimension of NetCDF: ',TRIM(infile)
  DO i=1,nr_dims
    CALL check(nf90_inquire_dimension(ncid,i,dim_name(i),dim_len(i)))
    write(*,'(A,A,A,I7)') '  ',TRIM(dim_name(i)), ':', dim_len(i)
  ENDDO
  CALL check(nf90_close(ncid))
END SUBROUTINE get_nc_dims

SUBROUTINE read_nc_1d(infile,varname,idata,nx)
  use iso_c_binding, only : c_float
  use netcdf
  REAL(c_float), DIMENSION(nx), INTENT(OUT) :: idata
  INTEGER(KIND=4), INTENT(IN) :: nx
  INTEGER(KIND=4), DIMENSION(1) :: dimids
  INTEGER(KIND=4) :: ncid, ndims, varid
  CHARACTER(LEN=*), INTENT(IN) :: infile, varname
  write(*,'(a)') 'read_nc_1d: ' // TRIM(infile) // ' -> ' // TRIM(varname)
  ! write(*,'(A)')     ' infile  : ' // TRIM(infile)
  ! write(*,'(A)')     ' varname : ' // TRIM(varname)
  ! write(*,'(A,I6)')  ' nx      : ', nx
  ! Get the values of the coordinates
  CALL check(nf90_open(TRIM(infile), nf90_nowrite, ncid))
  CALL check(nf90_inq_varid(ncid,TRIM(varname),varid))
  CALL check(nf90_inquire_variable(ncid=ncid,varid=varid,ndims=ndims,dimids=dimids))
  ! write(*,'(A,I6)')  ' ndims   : ', ndims
  ! write(*,'(A,1I6)') ' dimids  : ', dimids
  ! write(*,'(a)') ''
  ! Get the variable
  CALL check(nf90_get_var(ncid,varid,idata))
  CALL check(nf90_close(ncid))
END SUBROUTINE read_nc_1d

SUBROUTINE read_nc_2d(infile,varname,idata,nx,ny)
  use iso_c_binding, only : c_float
  use netcdf
  REAL(c_float), DIMENSION(nx,ny), INTENT(OUT) :: idata
  INTEGER(KIND=4), INTENT(IN) :: nx,ny
  INTEGER(KIND=4), DIMENSION(2) :: dimids
  INTEGER(KIND=4) :: ncid, ndims, varid
  CHARACTER(LEN=*), INTENT(IN) :: infile, varname
  write(*,'(a)') 'read_nc_2d: ' // TRIM(infile) // ' -> ' // TRIM(varname)
  ! write(*,'(A)')     ' infile  : ' // TRIM(infile)
  ! write(*,'(A)')     ' varname : ' // TRIM(varname)
  ! write(*,'(A,I6)')  ' nx      : ', nx
  ! write(*,'(A,I6)')  ' ny      : ', ny
  CALL check(nf90_open(TRIM(infile), nf90_nowrite, ncid))
  CALL check(nf90_inq_varid(ncid,TRIM(varname),varid))
  CALL check(nf90_inquire_variable(ncid=ncid,varid=varid,ndims=ndims,dimids=dimids))
  ! write(*,'(A,I6)')  ' ndims   : ', ndims
  ! write(*,'(A,2I6)') ' dimids  : ', dimids
  ! write(*,'(a)') ''
  ! Get the variable
  CALL check(nf90_get_var(ncid,varid,idata))
  CALL check(nf90_close(ncid))
END SUBROUTINE read_nc_2d

SUBROUTINE read_nc_3d(infile,varname,idata,nx,ny,nz)
  use iso_c_binding, only : c_float
  use netcdf
  REAL(c_float), DIMENSION(nx,ny,nz), INTENT(OUT) :: idata
  INTEGER(KIND=4), INTENT(IN) :: nx,ny,nz
  INTEGER(KIND=4), DIMENSION(3) :: dimids
  INTEGER(KIND=4) :: ncid, ndims, varid
  CHARACTER(LEN=*), INTENT(IN) :: infile, varname
  write(*,'(a)') 'read_nc_3d: ' // TRIM(infile) // ' -> ' // TRIM(varname)
  ! write(*,'(A)')     ' infile  : ' // TRIM(infile)
  ! write(*,'(A)')     ' varname : ' // TRIM(varname)
  ! write(*,'(A,I6)')  ' nx      : ', nx
  ! write(*,'(A,I6)')  ' ny      : ', ny
  ! write(*,'(A,I6)')  ' nz      : ', nz
  CALL check(nf90_open(TRIM(infile), nf90_nowrite, ncid))
  ! Get the values of the coordinates
  !CALL check(nf90_inquire_variable(ncid,1,vname,xtype,ndims,dimids))
  CALL check(nf90_inq_varid(ncid,TRIM(varname),varid))
  CALL check(nf90_inquire_variable(ncid=ncid,varid=varid,ndims=ndims))
  CALL check(nf90_inquire_variable(ncid=ncid,varid=varid,ndims=ndims,dimids=dimids))
  ! write(*,'(A,I6)')  ' ndims   : ', ndims
  ! write(*,'(A,3I6)') ' dimids  : ', dimids
  ! write(*,'(a)') ''
  CALL check(nf90_get_var(ncid,varid,idata))
  CALL check(nf90_close(ncid))
END SUBROUTINE read_nc_3d

SUBROUTINE mean_2d(input,mean,nx,ny)
  use iso_c_binding, only : c_float
  INTEGER, INTENT(IN) :: nx,ny
  REAL(c_float), INTENT(IN) :: input(nx,ny)
  REAL(c_float), INTENT(OUT):: mean
  INTEGER :: i,j
  mean = 0.0
  DO i=1,nx
    DO j=1,ny
      mean = mean + input(i,j)
    ENDDO
  ENDDO
  mean = mean / REAL(nx*ny)
END SUBROUTINE mean_2d

SUBROUTINE check(istatus)
  use netcdf
  INTEGER, INTENT (IN) :: istatus
  IF (istatus /= nf90_noerr) THEN
    write(*,'(A)') 'ERROR: '//TRIM((nf90_strerror(istatus)))
    call ABORT()
  END IF
END SUBROUTINE check
