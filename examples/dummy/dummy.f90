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

program resnet_forward
    use torch_ftn
    use iso_fortran_env

    implicit none

    integer, parameter :: rows = 3
    integer, parameter :: cols = 4
    integer, parameter :: depth = 2

    integer :: n
    type(torch_module) :: torch_mod
    type(torch_tensor_wrap) :: input_tensors
    type(torch_tensor) :: out_tensor

    real(real32) :: x2d(rows, cols)
    real(real32) :: x3d(rows, cols, depth)

    real(real32), pointer :: y3d(:, :, :)

    character(:), allocatable :: filename
    integer :: arglen, stat

    ! Hardcoded values for 2D array
    data x2d /  1.0,  2.0,  3.0,  4.0, &
                   5.0,  6.0,  7.0,  8.0, &
                   9.0, 10.0, 11.0, 12.0 /
    
    ! Hardcoded values for 3D array
    data x3d /  1.0,  2.0,  3.0,  4.0, &
                   5.0,  6.0,  7.0,  8.0, &
                   9.0, 10.0, 11.0, 12.0, &
                  13.0, 14.0, 15.0, 16.0, &
                  17.0, 18.0, 19.0, 20.0, &
                  21.0, 22.0, 23.0, 24.0 /
    

    if (command_argument_count() /= 1) then
        print *, "Need to pass a single argument: Pytorch model file name"
        stop
    end if

    call get_command_argument(number=1, length=arglen)
    allocate(character(arglen) :: filename)
    call get_command_argument(number=1, value=filename, status=stat)

    call input_tensors%create
    call input_tensors%add_array(x2d)
    call input_tensors%add_array(x3d)
    call torch_mod%load(filename)
    call torch_mod%forward(input_tensors, out_tensor)
    call out_tensor%to_array(y3d)

    print *, y3d(:, 1, 4)
end program
