program performance

  USE mpi
  implicit none

  INTEGER :: total_size,n_iterations,i,data_chunk,divide_by,mpi_rank
  INTEGER,ALLOCATABLE :: nproma(:)
  REAL,ALLOCATABLE :: time(:)

  character(1024) :: model_path

  INTEGER:: mpi_err,mpi_size

  CALL MPI_INIT(mpi_err)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, mpi_rank, mpi_err)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, mpi_size, mpi_err)

  total_size = 5000
  divide_by = 10
  n_iterations = 5

  ALLOCATE(nproma(divide_by))
  ALLOCATE(time(divide_by))

  data_chunk = total_size/divide_by

  ! Get CL arguments
  CALL get_command_argument(1, model_path)

  DO i=1,divide_by
    IF(i < divide_by)THEN
      nproma(i) = 64 +(i-1)*data_chunk
    ELSE
      nproma(i) = total_size
    ENDIF
      CALL perform_test(nproma(i),total_size,n_iterations,model_path,time(i))
  ENDDO

  CALL MPI_BARRIER(MPI_COMM_WORLD, mpi_err)

  CALL sleep(mpi_rank)
  WRITE(*,'(A)') '+++++++++++++++++++++++++++++++++++++++++++++++++++'
  DO i=1,divide_by
    WRITE(*,'(A,I4)') 'Timings for process: ', mpi_rank
    WRITE(*,'(A,I6,A,F4.1)') '     nproma: ',nproma(i),' total_time: ',time(i)
  ENDDO
  WRITE(*,'(A)') ''

  DEALLOCATE(nproma)
  DEALLOCATE(time)

  CALL MPI_FINALIZE(mpi_err)

CONTAINS

  SUBROUTINE perform_test(nproma,total_size,n_iterations,model_path,time)

    use inferof
    use iso_c_binding, only : c_float, c_null_char

    INTEGER,INTENT(IN) :: nproma,total_size,n_iterations
    character(1024),INTENT(IN) :: model_path
    REAL,INTENT(OUT) :: time

    character(1024) :: model_type='tf_c'
    character(1024) :: yaml_config

    ! model of infero model
    type(infero_model) :: model
    type(infero_tensor_set) :: iset
    type(infero_tensor_set) :: oset

    ! input and output tensors
    real(c_float) :: input_3d(nproma,60,1,6)
    real(c_float) :: input_2d(nproma,1,8)
    real(c_float) :: pred_flx(nproma,60,4)

    integer :: block,n_blocks,it
    REAL :: t_start,t_finish


    ! SETUP INFERO AND ASSIGN IN/OUT TENSORS

    ! init infero
    call infero_check(infero_initialise())

    ! YAML config string -> LW/SW
    yaml_config = "---"//NEW_LINE('A') &
      //"  path: "//TRIM(model_path)//NEW_LINE('A') &
      //"  type: "//TRIM(model_type)//c_null_char

    ! get a infero model for LW/SW
    call infero_check(model%initialise_from_yaml_string(yaml_config))


    ! init tensor sets
    call infero_check(iset%initialise())
    call infero_check(oset%initialise())

    ! input is identical for both models
    call infero_check(iset%push_tensor(input_3d, "serving_default_input_22"))
    call infero_check(iset%push_tensor(input_2d, "serving_default_input_23"))

    call infero_check(oset%push_tensor(pred_flx, "StatefulPartitionedCall"))

    n_blocks = total_size/nproma

    CALL random_number(input_3d)
    CALL random_number(input_2d)
    pred_flx(:,:,:) = 0.0_c_float

    CALL cpu_time(t_start)
    DO it=1,n_iterations
      DO block=1,n_blocks
        ! apply model
        call infero_check(model%infer(iset, oset ))
      ENDDO
    ENDDO
    CALL cpu_time(t_finish)
    time = t_finish - t_start


    ! CLEANUP 

    ! free the model
    call infero_check(model%free())

    ! finalise infero library
    call infero_check(infero_finalise())

  END SUBROUTINE perform_test
end program performance
