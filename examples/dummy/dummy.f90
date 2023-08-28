! Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! MIT License
! 
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.

program dummy
    use torch_ftn
    use iso_fortran_env

    implicit none

    integer, parameter :: num_var_2d   = 8
    integer, parameter :: num_var_3d   = 6
    integer, parameter :: feats_out_3d = 4
    integer, parameter :: len_height   = 70 ! number of vertical levels
    integer, parameter :: minibatch    = 1  ! number of simultaneous samples
                                            ! to make inference for

    integer :: n
    type(torch_module) :: torch_mod
    type(torch_tensor_wrap) :: input_tensors
    type(torch_tensor) :: out_tensor

    real(real32) :: x2d(minibatch, num_var_2d)
    real(real32) :: x3d(minibatch, len_height, num_var_3d)

    real(real32), pointer :: y3d(:, :, :)

    character(:), allocatable :: filename
    integer :: arglen, stat

    if (command_argument_count() /= 1) then
        print *, "Need to pass a single argument: Pytorch model file name"
        stop
    end if

    call get_command_argument(number=1, length=arglen)
    allocate(character(arglen) :: filename)
    call get_command_argument(number=1, value=filename, status=stat)

    ! Filling x2d with ascending integer numbers
    do n = 1, num_var_2d
        x2d(:, n) = n
    end do

    ! Filling x3d with ascending integer numbers
    do n = 1, num_var_3d
        x3d(:, :, n) = n
    end do

    call input_tensors%create
    call input_tensors%add_array(x2d)
    call input_tensors%add_array(x3d)
    call torch_mod%load(filename)
    call torch_mod%forward(input_tensors, out_tensor)
    call out_tensor%to_array(y3d)

    print *, y3d(:, 1, 4)
end program
