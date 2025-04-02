## Day 2 Learnings (Shared Memory Tiling):
Nvidia GPUs have physical CUDA cores, ofcourse.
But, from CUDA programming model's perspective these cores are divided into blocks and threads(there are other heirarchies, but for now lets limit the conversation).
The thread is the kernel we write. Simple as that. A block is group of threads such that they can be designed to share some resources.
One such resource is the shared memory,  **\_\_shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]** in code.

One of the main ideas in CUDA is combating memory bottlenecks, i.e. reduce the read/write operations from global memory.
That is where the shared memory comes in.

### Design of the kernel
The code is multiplying 2 [64 by 64] matrices. So the output matrix dim is [64 by 64].
Each kernel is computing 1 value in the output matrix. To do that, it must use 1 row from **mat_A** and 1 col from **mat_B**. This is where it gets interesting.

It is easy to see that 1 row from A is involved in the calculation of lots of output values, specifically 64 output values.
For e.g., row 1(with 64 values) of A is multiplied with each column of B(64 columns) to produce the row 1(that has 64 values) of the output matrix. And each of these values in the row 1 of output matrix is computed by a separate CUDA thread. As such, each of these threads would load the row1 of A separately during their routine.

But if we allocate a shared memory region, and load the row1 of A into it and run all the threads that use row1 of A in their calculation, we would save immensely on global memory reads.

*Line 21*

**\_\_shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; \
    \_\_shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];**

A problem that arises then is that the size of these rows might be large and they would not fit into the shared memory. So we break the row into TILE sized chuncks and use a **for loop** to go over the entire row.

*Line 39*\
**for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)**

Lastly, the values of the TILED rows of A & cols of B are loaded into the shared memory value-by-value by each thread, i.e. each thread only writes one value to populate the 32 by 32 shared memory region. 
We use **__syncthreads()** to ensure that all threads perform this population and continue execution.
So we have introduced some manual synchonization. That is cumbersome when the project grows, I guess.
Anyway, each thread would use the row and col needed for their output value and perform their multiplication at line 57
For a single output value we need to execute 2 TILES from A and B, so the for loop runs 2 times.
We then write the accumulated sum into the output matrix.

That's that.

Apparently optimization of this entire operation is so crucial that people have built a hardware core for it. I think that's what Tensor cores are.

See you tomorrow.