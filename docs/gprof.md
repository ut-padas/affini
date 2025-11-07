## Generation of Nsight Systems Profile for GPU Adaptive Clustering
```sh
c608-142[gh](651)$ nsys profile --trace=cuda,nvtx,osrt --output=gpu_adaptive_clustering_profile --force-overwrite true uv run python src/gpu_adaptive_clustering_new.py
Collecting data...
ABS_TOL=0.01, REL_TOL=0.01
N_INIT_CLUSTERS=100, MIN_PTS_PER_CLUSTER=50
Running on GPU. N_SAMPLES=10,000
  --> 98/100 clusters need splitting
Level 1: processing 196 clusters (prev level took 5.65s)
  --> 177/196 clusters need splitting
Level 2: processing 354 clusters (prev level took 0.75s)
  --> 179/354 clusters need splitting
Level 3: processing 358 clusters (prev level took 1.01s)
  --> 51/358 clusters need splitting
Level 4: processing 102 clusters (prev level took 0.61s)
  --> 7/102 clusters need splitting
Level 5: processing 14 clusters (prev level took 0.16s)
  --> 1/14 clusters need splitting
Level 6: processing 2 clusters (prev level took 0.03s)
  --> 0/2 clusters need splitting
Total final clusters formed: 613
Adaptive clustering completed in 9.54 seconds.
Total clusters violating tolerance on test points: (190 / 613) = 31.00%
Average cluster size: 26.67 points per cluster
Generating '/tmp/nsys-report-1d92.qdstrm'
[1/1] [========================100%] gpu_adaptive_clustering_profile.nsys-rep
Generated:
        /home1/10973/prajwal_prathiksh/github-repos/knnsur/gpu_adaptive_clustering_profile.nsys-rep
```

## Analysis of Nsight Systems Profile
```sh
c608-142[gh](652)$ nsys stats gpu_adaptive_clustering_profile.nsys-rep
NOTICE: Existing SQLite export found: gpu_adaptive_clustering_profile.sqlite
        It is assumed file was previously exported from: gpu_adaptive_clustering_profile.nsys-rep
        Consider using --force-export=true if needed.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances     Avg (ns)         Med (ns)        Min (ns)       Max (ns)      StdDev (ns)    Style                         Range                       
 --------  ---------------  ---------  ---------------  ---------------  -------------  -------------  -------------  -------  --------------------------------------------------
     48.6    1,108,079,136          3    369,359,712.0      2,145,920.0        225,568  1,105,707,648  637,696,741.5  PushPop  cuml_python:common.CumlArray.full                 
     48.5    1,105,714,208          1  1,105,714,208.0  1,105,714,208.0  1,105,714,208  1,105,714,208            0.0  PushPop  cuml_python:common.CumlArray.ones                 
      1.5       33,729,920      2,364         14,268.2         11,616.0          7,680         42,272        5,460.6  PushPop  CCCL:cub::DeviceReduce::Reduce                    
      1.1       24,104,800      2,254         10,694.2         10,496.0          7,680         37,312        1,690.6  PushPop  CCCL:cub::DeviceReduce::Sum                       
      0.2        5,274,304        513         10,281.3         10,048.0          8,224         40,192        1,742.0  PushPop  CCCL:cub::DeviceRadixSort                         
      0.1        2,384,352          2      1,192,176.0      1,192,176.0        231,232      2,153,120    1,358,980.0  PushPop  cuml_python:common.CumlArray.zeros                
      0.0        1,016,256        106          9,587.3          7,888.0          6,912         48,864        6,008.8  PushPop  CCCL:cub::DeviceScan::InclusiveSum                
      0.0          391,200         99          3,951.5          3,680.0          3,168         23,520        2,024.1  PushPop  CCCL:cub::DeviceReduce::ArgMin                    
      0.0          230,112          4         57,528.0         52,432.0         28,544         96,704       30,129.1  PushPop  cuml_python:internals.CumlArray.__init__          
      0.0          223,104          1        223,104.0        223,104.0        223,104        223,104            0.0  PushPop  cuml_python:common.input_utils.input_to_cuml_array
      0.0          222,432         52          4,277.5          3,680.0          3,136         18,752        2,500.5  PushPop  CCCL:cub::DeviceFor::Bulk                         
      0.0          202,656          1        202,656.0        202,656.0        202,656        202,656            0.0  PushPop  cuml_python:common.CumlArray.from_input           
      0.0          171,136          7         24,448.0         17,792.0         13,696         46,624       13,392.1  PushPop  CCCL:cub::DeviceReduce::Max                       
      0.0           82,336          8         10,292.0          7,504.0          6,848         30,080        8,009.2  PushPop  CCCL:cub::DeviceSelect::If                        
      0.0           55,936          8          6,992.0          5,072.0          4,768         14,240        3,781.3  PushPop  CCCL:cub::DeviceFor::ForEachN                     
      0.0           44,320          1         44,320.0         44,320.0         44,320         44,320            0.0  PushPop  cuml_python:common.CumlArray.to_output            
      0.0           24,032          1         24,032.0         24,032.0         24,032         24,032            0.0  PushPop  CCCL:cub::DeviceHistogram::MultiHistogramEven     

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls      Avg (ns)          Med (ns)         Min (ns)        Max (ns)       StdDev (ns)             Name         
 --------  ---------------  ---------  ----------------  ----------------  --------------  --------------  ---------------  ----------------------
     26.3   17,833,120,800          1  17,833,120,800.0  17,833,120,800.0  17,833,120,800  17,833,120,800              0.0  pthread_join          
     23.6   15,998,795,136          9   1,777,643,904.0         318,336.0          10,720  15,791,710,816  5,255,711,841.7  epoll_wait            
     19.2   12,985,643,296        141      92,096,761.0     100,103,808.0           1,920     100,117,984     26,007,206.3  poll                  
     15.8   10,689,645,504         15     712,643,033.6          19,680.0           1,184  10,000,126,336  2,575,288,334.0  futex                 
      9.0    6,113,208,672         10     611,320,867.2     669,275,136.0           1,120   1,308,854,688    479,698,917.0  fcntl                 
      2.9    1,974,502,304          6     329,083,717.3     319,616,512.0           1,760     669,608,128    307,317,252.9  flock                 
      1.8    1,222,531,264     60,534          20,195.8          15,472.0           1,024      15,239,072         87,794.1  ioctl                 
      1.2      788,291,200      3,273         240,846.7         217,824.0           1,184      16,750,560        375,393.6  open64                
      0.1       57,133,792      2,566          22,265.7           3,584.0           1,024      16,942,912        439,470.2  read                  
      0.0       13,556,608      6,603           2,053.1           1,632.0           1,088         300,256          5,918.0  munmap                
      0.0        5,730,112         12         477,509.3           7,296.0           3,360       2,007,520        748,267.7  fread                 
      0.0        4,771,840         45         106,040.9           4,224.0           1,728       1,183,200        256,138.3  fopen                 
      0.0        4,716,192      2,457           1,919.5           1,888.0           1,024           7,168            486.1  close                 
      0.0        4,673,120      1,988           2,350.7           1,568.0           1,024          22,048          2,014.3  mmap64                
      0.0        2,308,320        110          20,984.7           1,472.0           1,024         630,752         85,966.5  statx                 
      0.0          524,352          9          58,261.3           2,944.0           2,400         267,872        106,415.6  fopen64               
      0.0          452,992         49           9,244.7           7,296.0           1,504          45,920          8,056.4  mmap                  
      0.0          413,792          1         413,792.0         413,792.0         413,792         413,792              0.0  posix_spawnp          
      0.0          387,232          4          96,808.0          97,728.0          84,224         107,552          9,960.5  pthread_create        
      0.0          226,368          9          25,152.0          16,512.0           4,992         102,720         29,606.0  sem_timedwait         
      0.0          150,720          2          75,360.0          75,360.0          29,088         121,632         65,438.5  waitpid               
      0.0          108,160          5          21,632.0          24,320.0          12,544          28,320          6,325.5  getline               
      0.0           93,920         35           2,683.4           2,144.0           1,024           7,360          1,811.1  fclose                
      0.0           77,024         12           6,418.7           5,744.0           4,992           9,696          1,563.1  pthread_cond_signal   
      0.0           76,096          3          25,365.3          28,576.0          18,208          29,312          6,209.3  pthread_rwlock_wrlock 
      0.0           66,592          7           9,513.1           5,280.0           1,056          38,176         13,236.1  fgets                 
      0.0           63,072          2          31,536.0          31,536.0          27,840          35,232          5,226.9  pthread_rwlock_rdlock 
      0.0           57,120          1          57,120.0          57,120.0          57,120          57,120              0.0  pthread_cond_wait     
      0.0           54,400         20           2,720.0           2,336.0           1,216           9,888          1,864.7  write                 
      0.0           49,440         11           4,494.5           4,896.0           1,408           8,352          2,474.2  open                  
      0.0           24,032          5           4,806.4           4,960.0           2,240           7,616          2,219.4  pipe2                 
      0.0           15,136          2           7,568.0           7,568.0           6,368           8,768          1,697.1  socket                
      0.0           11,360          4           2,840.0           3,296.0           1,184           3,584          1,131.8  epoll_ctl             
      0.0            8,320          1           8,320.0           8,320.0           8,320           8,320              0.0  connect               
      0.0            8,192          1           8,192.0           8,192.0           8,192           8,192              0.0  socketpair            
      0.0            7,264          4           1,816.0           1,472.0           1,312           3,008            798.2  mprotect              
      0.0            5,312          3           1,770.7           1,536.0           1,216           2,560            702.1  signal                
      0.0            4,224          2           2,112.0           2,112.0           1,792           2,432            452.5  fwrite                
      0.0            3,040          2           1,520.0           1,520.0           1,280           1,760            339.4  sigaction             
      0.0            2,688          1           2,688.0           2,688.0           2,688           2,688              0.0  pthread_cond_broadcast
      0.0            1,440          1           1,440.0           1,440.0           1,440           1,440              0.0  bind                  
      0.0            1,024          1           1,024.0           1,024.0           1,024           1,024              0.0  dup2                  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)    Max (ns)   StdDev (ns)                        Name                      
 --------  ---------------  ---------  -----------  -----------  ---------  ----------  -----------  ------------------------------------------------
     42.5    1,282,448,096     70,540     18,180.4      5,696.0      1,216  98,139,200    370,689.3  cudaMalloc                                      
     30.1      908,075,840     70,546     12,872.1      5,632.0        512   1,071,872     20,512.6  cudaFree                                        
     13.5      407,823,168     60,286      6,764.8      5,888.0      2,016      83,200      3,272.8  cuLaunchKernel                                  
      4.8      145,857,056     22,130      6,590.9      4,960.0      1,952   3,516,768     26,497.1  cudaLaunchKernel                                
      4.5      136,642,336     10,966     12,460.5     12,480.0         32   3,289,792     31,451.5  cudaMemcpyAsync                                 
      1.1       33,615,648         80    420,195.6    101,952.0     83,456   1,534,400    545,681.9  cuModuleLoadData                                
      0.7       21,707,744         47    461,866.9     90,016.0     48,544   1,059,296    442,485.8  cuModuleUnload                                  
      0.5       14,835,968     10,736      1,381.9      1,056.0        448      16,224      1,694.9  cudaStreamSynchronize                           
      0.5       14,342,464      1,844      7,777.9      8,448.0      2,752      23,872      2,891.8  cudaLaunchKernelExC_v11060                      
      0.3       10,408,576        927     11,228.2      2,784.0      1,952     895,104     37,817.9  cudaStreamCreateWithFlags                       
      0.3        8,606,432      1,544      5,574.1      2,400.0      1,792   2,885,696     81,651.9  cudaLaunchKernel_ptsz                           
      0.3        8,376,192     15,579        537.7        480.0        192      10,560        234.6  cudaStreamIsCapturing_v10000                    
      0.2        6,912,640        636     10,868.9     10,832.0      4,288      27,808      1,659.6  cudaMemsetAsync                                 
      0.1        3,235,328        928      3,486.3      3,008.0      1,760      36,608      2,760.4  cudaStreamDestroy                               
      0.1        3,105,792          2  1,552,896.0  1,552,896.0  1,095,648   2,010,144    646,646.3  cudaGetDeviceProperties_v2_v12000               
      0.1        2,696,256      5,400        499.3        384.0        160     192,256      2,628.9  cuKernelGetFunction                             
      0.1        1,579,040        577      2,736.6      2,720.0        704       6,048        534.7  cudaEventQuery                                  
      0.0        1,420,960         15     94,730.7     99,968.0     41,344     166,752     38,696.0  cuLibraryLoadData                               
      0.0        1,122,912        550      2,041.7      1,952.0      1,568       9,536        640.8  cudaEventRecord                                 
      0.0          644,800      4,620        139.6         96.0         32       1,024        116.0  cuGetProcAddress_v2                             
      0.0          537,664          4    134,416.0      6,352.0      4,352     520,608    257,467.0  cudaHostAlloc                                   
      0.0          351,904      3,113        113.0         96.0         32       6,176        168.9  cuStreamGetCaptureInfo_v2                       
      0.0          286,144          1    286,144.0    286,144.0    286,144     286,144          0.0  cuLibraryUnload                                 
      0.0          259,776        128      2,029.5      2,016.0        992       8,960        969.8  cudaMemsetAsync_ptsz                            
      0.0           91,008        122        746.0        608.0        160       4,992        635.6  cudaEventDestroy                                
      0.0           89,344        120        744.5        288.0        192       8,960      1,046.6  cudaEventCreateWithFlags                        
      0.0           57,600         20      2,880.0      1,584.0        736      12,608      2,783.2  cudaDeviceSynchronize                           
      0.0           30,656          1     30,656.0     30,656.0     30,656      30,656          0.0  cudaMemGetInfo                                  
      0.0           22,080         14      1,577.1      1,552.0        800       2,368        448.6  cuInit                                          
      0.0           13,376         35        382.2        288.0        224       1,888        363.1  cudaOccupancyMaxActiveClusters_v11070           
      0.0           11,264         16        704.0        496.0        384       1,920        514.0  cuLibraryGetKernel                              
      0.0            9,632          1      9,632.0      9,632.0      9,632       9,632          0.0  cudaStreamCreate                                
      0.0            4,448          1      4,448.0      4,448.0      4,448       4,448          0.0  cudaMemcpyAsync_ptsz                            
      0.0            4,384          2      2,192.0      2,192.0      2,112       2,272        113.1  cudaOccupancyAvailableDynamicSMemPerBlock_v10200
      0.0            4,096          3      1,365.3      1,024.0        832       2,240        763.5  cudaEventCreate                                 
      0.0            1,824         12        152.0         96.0         32         864        226.7  cuModuleGetLoadingMode                          
      0.0            1,088          2        544.0        544.0        384         704        226.3  cudaGetDriverEntryPoint_v11030                  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     21.9       76,864,992      1,853   41,481.4   61,696.0     1,376    62,208     28,500.9  _norm_ord2                                                                                          
      9.2       32,116,319      1,842   17,435.6   17,408.0    16,672    18,816        330.6  void execute_dist<interval_32_functor, curand_pseudo_state<curandStateXORWOW>, int, unsigned int, u…
      8.0       27,876,352        613   45,475.3   45,472.0    45,312    45,760         83.8  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize64x128x32_stage3_warpsize2x4x1_tensor16x8x16_aligna8_…
      7.6       26,459,392      1,535   17,237.4   17,216.0    16,416    18,464        342.3  void execute_dist<random_uniform_functor, curand_pseudo_state<curandStateXORWOW>, double, >(long, l…
      6.1       21,411,456     10,275    2,083.8    2,240.0       928     4,032        751.6  cupy_copy__float64_float64                                                                          
      5.4       18,981,024      1,126   16,857.0   17,120.0    14,720    18,880        875.5  void execute_dist<standard_normal_functor, curand_pseudo_state<curandStateXORWOW>, double, >(long, …
      4.9       17,219,200        513   33,565.7   37,088.0    13,824    37,824      7,162.2  sm90_xmma_gemm_f64f64_f64f64_f64_tt_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      2.4        8,488,768      3,348    2,535.5    2,528.0     2,112     3,328        231.1  void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_nn_align1>(T1::Params)                
      2.1        7,328,768      3,408    2,150.5    2,112.0       960     4,224        777.6  cupy_add__float64_float64_float64                                                                   
      1.9        6,687,648      2,993    2,234.4    2,016.0     1,024     3,776        722.0  cupy_subtract__float64_float64_float64                                                              
      1.8        6,377,792        513   12,432.3   12,448.0    12,224    12,736        109.8  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceRadixSortS…
      1.7        6,089,472      3,581    1,700.5    1,312.0       992     3,616        807.4  cupy_multiply__float64_float64_float64                                                              
      1.7        5,883,680      1,026    5,734.6    5,760.0     5,440     6,144        122.1  void geqr2_smem<double, double, (int)8, (int)5, (int)5>(int, int, T1 *, unsigned long, T1 *, int)   
      1.3        4,722,080      2,052    2,301.2    2,368.0     2,048     2,816        152.3  void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_tn_align1>(T1::Params)                
      1.2        4,376,704      2,487    1,759.8    1,888.0       992     2,144        354.3  cupy_multiply__float_float64_float64                                                                
      1.1        3,736,384      1,254    2,979.6    3,008.0     1,568     3,232        177.4  cupy_power__float64_float_float64                                                                   
      1.1        3,709,824      2,170    1,709.6    1,728.0     1,408     2,208        145.2  cupy_scan_naive                                                                                     
      1.0        3,562,880      1,963    1,815.0    1,888.0     1,536     2,656        169.9  cupy_take                                                                                           
      0.9        3,234,368      2,252    1,436.2    1,440.0     1,184     1,760         99.6  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.9        3,227,008      1,639    1,968.9    1,984.0     1,696     2,624        137.1  cupy_mean                                                                                           
      0.8        2,973,792      3,078      966.1      992.0       736     1,536        120.3  copy_info_kernel(int, int *)                                                                        
      0.8        2,965,824      1,881    1,576.7    1,600.0       800     1,728         87.7  cupy_fill                                                                                           
      0.8        2,906,368      1,126    2,581.1    3,520.0     1,184     3,872      1,070.2  cupy_true_divide__float64_float64_float64                                                           
      0.8        2,821,888      1,739    1,622.7    1,632.0     1,344     2,272        138.9  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.8        2,779,360      1,539    1,806.0    1,824.0     1,600     2,208        108.3  void larft_T_32<double, (int)256>(int, T1 *, int, long, const T1 *, long)                           
      0.7        2,419,296        513    4,716.0    4,704.0     4,512     5,152         87.5  sm90_xmma_gemm_f64f64_f64f64_f64_tt_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      0.7        2,300,896      1,539    1,495.1    1,536.0     1,312     1,824        101.9  void larft_vtv_32<double, (int)128>(long, long, const T1 *, long, long, T1 *, long, long)           
      0.6        2,191,392      1,842    1,189.7    1,184.0       992     1,472         98.1  cupy_add__int_uint32_int64                                                                          
      0.6        2,166,144        613    3,533.7    3,552.0     3,264     3,872        106.2  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceKern…
      0.6        2,069,312      1,539    1,344.6    1,312.0       960     1,728        186.8  cupy_copy__bool_bool                                                                                
      0.5        1,749,984      1,032    1,695.7    1,760.0     1,408     2,048        144.8  cupy_getitem_mask                                                                                   
      0.5        1,717,728          3  572,576.0  572,512.0   572,192   573,024        419.7  void init_curand<curand_pseudo_state<curandStateXORWOW>>(long, unsigned long, long)                 
      0.5        1,614,752      1,126    1,434.1    1,472.0     1,184     2,016        151.2  cupy_sqrt__float64_float64                                                                          
      0.4        1,514,880      1,126    1,345.4    1,328.0     1,088     1,632        148.3  cupy_true_divide__float64_float_float64                                                             
      0.4        1,462,080      1,026    1,425.0    1,440.0     1,152     1,824        148.1  cupy_triu_kernel                                                                                    
      0.4        1,331,712      1,026    1,298.0    1,280.0     1,088     1,472         97.7  cupy_cub_sum_pass2                                                                                  
      0.4        1,312,320      1,026    1,279.1    1,280.0     1,088     1,472         97.8  cupy_cub_sum_pass1                                                                                  
      0.4        1,278,368      1,147    1,114.5    1,120.0       928     1,408        100.0  cupy_copy__int64_int64                                                                              
      0.4        1,277,536      1,026    1,245.2    1,232.0     1,088     1,376         94.3  void lacpy_kernel<double, double, (int)5, (int)3>(long, long, const T1 *, long, long, T2 *, long, l…
      0.4        1,265,056      1,126    1,123.5    1,120.0       896     1,280         97.4  cupy_copy__int32_int32                                                                              
      0.3        1,201,312        921    1,304.4    1,312.0     1,120     1,472         96.3  cupy_remainder__int64_int_int64                                                                     
      0.3        1,185,344        513    2,310.6    2,304.0     2,016     2,624        121.4  std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (b…
      0.3        1,148,448        613    1,873.5    1,856.0     1,632     2,336        122.8  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.3        1,139,360        627    1,817.2    1,824.0     1,024     1,984         85.9  cupy_negative__float64_float64                                                                      
      0.3        1,111,200      1,026    1,083.0    1,056.0       896     1,216         94.8  void batch_eye_kernel<double, (int)5, (int)3>(long, long, T1 *, unsigned long, long long, int)      
      0.3        1,099,456        921    1,193.8    1,216.0     1,024     1,440         94.8  cupy_add__int64_int64_int64                                                                         
      0.3        1,077,728        921    1,170.2    1,152.0       960     1,312         96.6  cupy_subtract__float_float64_float64                                                                
      0.3          959,328      1,026      935.0      928.0       736     1,248        102.2  void orgqr_step1_kernel<double, (int)5, (int)3>(int, int, int, T1 *, int)                           
      0.2          727,264        613    1,186.4    1,248.0     1,056     1,408         94.9  cupy_greater__float64_float_bool                                                                    
      0.2          708,800        513    1,381.7    1,344.0     1,184     1,536         97.8  cupy_cub_mean_pass2                                                                                 
      0.2          679,776        513    1,325.1    1,344.0     1,120     1,472         99.9  cupy_cub_mean_pass1                                                                                 
      0.2          676,896         42   16,116.6   15,136.0    11,648    92,544     12,175.5  void cutlass_raft_cutlass::Kernel<cuvs::gemm::kernel::FusedDistanceNNPersistent<cutlass_raft_cutlas…
      0.2          675,040        513    1,315.9    1,344.0     1,024     1,920        110.4  void larfb_set_v1_R_kernel<double, (int)16>(long, T1 *, long, long, T1 *, long, long)               
      0.2          673,920        513    1,313.7    1,280.0     1,120     1,536         95.5  _exists_nan_pass1                                                                                   
      0.2          666,592        513    1,299.4    1,312.0     1,088     1,440         95.6  _exists_nan_pass2                                                                                   
      0.2          632,288         99    6,386.7    6,368.0     6,240     6,816        114.5  void cutlass_raft_cutlass::Kernel2<cutlass_raft_cutlass::gemm::kernel::GemmWithFusedEpilogue<cutlas…
      0.2          621,696        513    1,211.9    1,184.0     1,056     1,472         96.3  void larfb_restore_v1_R_kernel<double, (int)16>(long, T1 *, long, long, T1 *, long, long)           
      0.2          614,336        513    1,197.5    1,152.0     1,024     1,344         95.7  cupy_where__bool_float_float64_float64                                                              
      0.2          610,848        520    1,174.7    1,184.0       992     1,760        106.4  cupy_greater__float64_float64_bool                                                                  
      0.2          599,040        513    1,167.7    1,120.0       992     1,408         94.1  cupy_less_equal__float64_float64_bool                                                               
      0.1          518,048        147    3,524.1    3,520.0     3,072     4,160        230.5  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      0.1          398,432         10   39,843.2   36,912.0    10,240    79,104     26,503.6  void cutlass_raft_cutlass::Kernel<cuvs::gemm::kernel::FusedDistanceNNPersistent<cutlass_raft_cutlas…
      0.1          373,312        255    1,464.0    1,472.0     1,248     1,888        119.1  void raft::linalg::detail::coalescedSumThinKernel<raft::linalg::detail::ReductionThinPolicy<(int)4,…
      0.1          343,040         99    3,465.1    3,360.0     3,264     3,648        141.1  void raft::random::detail::sample_with_replacement_kernel<raft::random::detail::PCGenerator, int, d…
      0.1          288,896        246    1,174.4    1,120.0     1,056     1,312         94.8  cupy_less__float64_float_bool                                                                       
      0.1          244,064         99    2,465.3    2,528.0     2,336     2,592         95.1  void raft::linalg::detail::coalescedSumMediumKernel<(int)256, double, double, int, raft::identity_o…
      0.1          237,216         99    2,396.1    2,464.0     2,272     2,528         92.0  void cub::detail::scan::DeviceScanKernel<cub::detail::scan::policy_hub<double, double, double, unsi…
      0.1          221,056        100    2,210.6    2,208.0     2,112     2,336         45.9  cupy_nonzero_kernel_incomplete_scan                                                                 
      0.1          208,896         99    2,110.1    2,112.0     2,048     2,208         37.4  void raft::matrix::detail::matrixLinewiseVecRowsMainKernel<double, int, (unsigned long)16, (int)256…
      0.1          191,584        106    1,807.4    1,824.0     1,696     2,080         59.9  cupy_bsum_shfl                                                                                      
      0.1          189,888         58    3,273.9    3,296.0     2,592     3,680        277.0  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_k…
      0.0          145,504         99    1,469.7    1,536.0     1,344     1,568         92.9  void raft::matrix::detail::matrixLinewiseVecRowsTailKernel<double, int, (unsigned long)32, raft::mi…
      0.0          138,592         99    1,399.9    1,344.0     1,280     1,536         96.0  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<cub::KeyValu…
      0.0          137,760        100    1,377.6    1,376.0     1,344     1,568         31.6  cupy_equal__int32_int_bool                                                                          
      0.0          136,192         99    1,375.7    1,344.0     1,248     1,504         98.4  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0          117,344         38    3,088.0    3,104.0     2,784     3,616        176.3  void gemmSN_NN_kernel<double, (int)128, (int)2, (int)4, (int)8, (int)3, (int)4, (bool)0, cublasGemv…
      0.0          112,352         38    2,956.6    3,616.0     1,376     3,712      1,020.0  void raft::linalg::detail::sum_rows_by_key_large_nkeys_kernel_rowmajor<double *, thrust::transform_…
      0.0          110,816         99    1,119.4    1,184.0       928     1,312        143.1  void cub::detail::scan::DeviceScanInitKernel<cub::ScanTileState<double, (bool)1>>(T1, int)          
      0.0           77,728         52    1,494.8    1,504.0     1,152     1,632        121.5  void cuvs::distance::detail::initBinMutexKernel<int>(cuda::__4::counting_semaphore<(cuda::std::__4:…
      0.0           61,376         38    1,615.2    1,632.0     1,504     1,728         97.2  void raft::matrix::detail::matrixLinewiseVecColsMainKernel<double, int, (unsigned long)16, (int)256…
      0.0           58,976         38    1,552.0    1,552.0     1,376     1,728        140.9  void raft::linalg::detail::mapThenReduceKernel<double, double, unsigned long, raft::compose_op<raft…
      0.0           56,672         42    1,349.3    1,376.0     1,152     1,440         86.5  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, long, …
      0.0           55,680         26    2,141.5    2,144.0     2,080     2,208         33.8  void raft::linalg::detail::reduce_cols_by_key_cached_kernel<double, thrust::transform_iterator<cuvs…
      0.0           50,944          4   12,736.0   12,768.0     9,824    15,584      3,270.8  std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (b…
      0.0           44,800         38    1,178.9    1,120.0     1,056     1,312         94.0  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0           39,360          8    4,920.0    4,928.0     4,800     5,024         76.0  void raft::random::detail::rngKernel<(int)1, double, int, raft::random::detail::PCGenerator, raft::…
      0.0           29,024          8    3,628.0    3,568.0     3,200     3,936        258.2  void cub::detail::select::DeviceSelectSweepKernel<cub::detail::select::policy_hub<cub::KeyValuePair…
      0.0           20,480         10    2,048.0    2,080.0     1,504     2,656        333.6  cupy_concatenate_same_size                                                                          
      0.0           19,712         13    1,516.3    1,472.0     1,216     1,760        197.5  cupy_cub_any_pass1                                                                                  
      0.0           19,680          9    2,186.7    2,272.0     1,984     2,304        121.9  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0           18,816         13    1,447.4    1,440.0     1,248     1,632        137.7  cupy_cub_any_pass2                                                                                  
      0.0           18,464          7    2,637.7    2,560.0     2,304     3,040        246.4  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, double, double, double, double, (bool)0, …
      0.0           16,800         12    1,400.0    1,408.0     1,280     1,536        100.4  void raft::linalg::detail::reduce_cols_by_key_direct_kernel<double, thrust::transform_iterator<cuvs…
      0.0           15,904         11    1,445.8    1,504.0     1,312     1,536        101.0  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0           15,232          7    2,176.0    2,176.0     2,048     2,304         80.5  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceScanKernel…
      0.0           12,096          8    1,512.0    1,520.0     1,344     1,664        120.6  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0           11,744         10    1,174.4    1,184.0       928     1,312        105.6  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, long, …
      0.0           11,488          1   11,488.0   11,488.0    11,488    11,488          0.0  void cub::detail::histogram::DeviceHistogramSweepKernel<cub::detail::histogram::policy_hub<int, dou…
      0.0           10,272          6    1,712.0    1,536.0     1,312     2,464        470.6  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.0           10,144          7    1,449.1    1,472.0     1,088     1,728        210.4  cupy_maximum__float_float64_float64                                                                 
      0.0           10,016          6    1,669.3    1,744.0     1,408     1,920        201.8  cupy_nonzero_kernel                                                                                 
      0.0            9,888          8    1,236.0    1,232.0     1,120     1,376        104.0  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, int, c…
      0.0            9,248          8    1,156.0    1,168.0       992     1,312        151.5  void cub::detail::scan::DeviceCompactInitKernel<cub::ScanTileState<int, (bool)1>, int *>(T1, int, T…
      0.0            8,768          2    4,384.0    4,384.0     4,096     4,672        407.3  cupy_concatenate                                                                                    
      0.0            8,352          7    1,193.1    1,248.0       928     1,344        156.5  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceScanInitKe…
      0.0            7,936          8      992.0      976.0       864     1,152        110.9  cupy_arange__int_int_int64                                                                          
      0.0            7,616          6    1,269.3    1,184.0     1,120     1,728        233.1  cupy_bincount_kernel                                                                                
      0.0            7,328          6    1,221.3    1,216.0     1,088     1,376        117.0  cupy_less__int32_int_bool                                                                           
      0.0            7,264          6    1,210.7    1,184.0     1,184     1,280         42.5  cupy_greater__int64_int_bool                                                                        
      0.0            7,040          7    1,005.7    1,056.0       800     1,152        125.1  cupy_arange__int_int_int32                                                                          
      0.0            6,304          1    6,304.0    6,304.0     6,304     6,304          0.0  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (…
      0.0            4,736          2    2,368.0    2,368.0     2,240     2,496        181.0  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (…
      0.0            4,480          2    2,240.0    2,240.0     2,144     2,336        135.8  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0            4,448          2    2,224.0    2,224.0     2,176     2,272         67.9  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0            3,520          2    1,760.0    1,760.0     1,728     1,792         45.3  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0            3,040          2    1,520.0    1,520.0     1,504     1,536         22.6  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            2,560          2    1,280.0    1,280.0     1,248     1,312         45.3  cupy_copy__int_float64                                                                              
      0.0            2,336          1    2,336.0    2,336.0     2,336     2,336          0.0  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceKern…
      0.0            2,016          1    2,016.0    2,016.0     2,016     2,016          0.0  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0            1,600          1    1,600.0    1,600.0     1,600     1,600          0.0  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.0            1,568          1    1,568.0    1,568.0     1,568     1,568          0.0  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            1,408          1    1,408.0    1,408.0     1,408     1,408          0.0  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            1,216          1    1,216.0    1,216.0     1,216     1,216          0.0  cupy_copy__int_int32                                                                                
      0.0            1,024          1    1,024.0    1,024.0     1,024     1,024          0.0  void cub::detail::histogram::DeviceHistogramInitKernel<(int)1, double, int>(cuda::std::__4::array<i…

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     80.3       12,653,984  8,932   1,416.7   1,376.0     1,312     3,072        158.1  [CUDA memcpy Device-to-Host]  
     12.2        1,929,344  1,803   1,070.1   1,088.0       832     1,504        106.8  [CUDA memcpy Device-to-Device]
      6.7        1,052,672    764   1,377.8   1,408.0       704     4,800        243.7  [CUDA memset]                 
      0.7          113,888    117     973.4     896.0       704     1,376        204.7  [CUDA memcpy Host-to-Device]  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
    531.579    764     0.696     0.800     0.000    12.976        0.832  [CUDA memset]                 
      2.097  1,803     0.001     0.000     0.000     0.080        0.004  [CUDA memcpy Device-to-Device]
      0.065  8,932     0.000     0.000     0.000     0.001        0.000  [CUDA memcpy Device-to-Host]  
      0.045    117     0.000     0.000     0.000     0.010        0.001  [CUDA memcpy Host-to-Device]  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/openmp_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain OpenMP event data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/opengl_khr_range_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/opengl_khr_gpu_range_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/vulkan_marker_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/vulkan_gpu_marker_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/dx11_pix_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain DX11 CPU debug markers.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/dx12_gpu_marker_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain DX12 GPU debug markers.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/dx12_pix_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain DX12 CPU debug markers.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/wddm_queue_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain WDDM context data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/um_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/um_total_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/um_cpu_page_faults_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/openacc_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain OpenACC event data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/host-linux-armv8/reports/syscall_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain syscall data.
```