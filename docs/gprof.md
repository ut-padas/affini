## Generation of Nsight Systems Profile for GPU Adaptive Clustering
```sh
c608-082[gh](675)$ nsys profile --trace=cuda,nvtx,osrt --output=gpu_adaptive_clustering_profile --force-overwrite true uv run python src/gpu_adaptive_clustering_optimized.py
ABS_TOL=0.001, REL_TOL=0.001
Running on GPU. N_SAMPLES=10,000
  --> 100/100 clusters need splitting
Level 1: processing 200 clusters (prev level took 0.58s)
  --> 200/200 clusters need splitting
Level 2: processing 400 clusters (prev level took 0.86s)
  --> 400/400 clusters need splitting
Level 3: processing 800 clusters (prev level took 1.81s)
  --> 799/800 clusters need splitting
Level 4: processing 1598 clusters (prev level took 3.65s)
  --> 1504/1598 clusters need splitting
Level 5: processing 3008 clusters (prev level took 6.92s)
  --> 2061/3008 clusters need splitting
Level 6: processing 4122 clusters (prev level took 10.80s)
  --> 1568/4122 clusters need splitting
Level 7: processing 3136 clusters (prev level took 11.07s)
  --> 694/3136 clusters need splitting
Level 8: processing 1388 clusters (prev level took 7.18s)
  --> 274/1388 clusters need splitting
Level 9: processing 548 clusters (prev level took 3.18s)
  --> 81/548 clusters need splitting
Level 10: processing 162 clusters (prev level took 1.19s)
  --> 20/162 clusters need splitting
Level 11: processing 40 clusters (prev level took 0.36s)
  --> 5/40 clusters need splitting
Level 12: processing 10 clusters (prev level took 0.10s)
  --> 1/10 clusters need splitting
Level 13: processing 2 clusters (prev level took 0.04s)
  --> 1/2 clusters need splitting
Level 14: processing 2 clusters (prev level took 0.02s)
  --> 0/2 clusters need splitting
Total final clusters formed: 7808
Adaptive clustering completed in 48.02 seconds.
Generating '/tmp/nsys-report-7642.qdstrm'
[1/1] [========================100%] gpu_adaptive_clustering_profile.nsys-rep
Generated:
    /home1/10973/prajwal_prathiksh/github-repos/knnsur/gpu_adaptive_clustering_profile.nsys-rep
```

## Analysis of Nsight Systems Profile
```sh
c608-082[gh](677)$ nsys stats gpu_adaptive_clustering_profile.nsys-rep
Generating SQLite file gpu_adaptive_clustering_profile.sqlite from gpu_adaptive_clustering_profile.nsys-rep
Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range                    
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ---------------------------------------------
     45.5      264,755,872     23,336  11,345.4  11,136.0     8,064    43,744      1,530.7  PushPop  CCCL:cub::DeviceReduce::Reduce               
     39.9      232,156,064     23,226   9,995.5   9,824.0     6,976    59,808      1,594.7  PushPop  CCCL:cub::DeviceReduce::Sum                  
     14.1       82,108,608      7,708  10,652.4  10,464.0     8,288    45,056      1,352.5  PushPop  CCCL:cub::DeviceRadixSort                    
      0.2        1,404,224        114  12,317.8   9,328.0     8,416    47,008      7,840.6  PushPop  CCCL:cub::DeviceScan::InclusiveSum           
      0.1          731,424         15  48,761.6  21,792.0    11,648   427,488    105,226.3  PushPop  CCCL:cub::DeviceReduce::Max                  
      0.1          434,528         99   4,389.2   4,128.0     3,552    18,144      1,538.3  PushPop  CCCL:cub::DeviceReduce::ArgMin               
      0.0          255,328         52   4,910.2   4,304.0     3,488    20,192      2,778.2  PushPop  CCCL:cub::DeviceFor::Bulk                    
      0.0           99,392          8  12,424.0   9,184.0     8,864    34,432      8,906.7  PushPop  CCCL:cub::DeviceSelect::If                   
      0.0           52,480          8   6,560.0   5,344.0     4,672    15,520      3,640.7  PushPop  CCCL:cub::DeviceFor::ForEachN                
      0.0           26,624          1  26,624.0  26,624.0    26,624    26,624          0.0  PushPop  CCCL:cub::DeviceHistogram::MultiHistogramEven

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls      Avg (ns)          Med (ns)         Min (ns)        Max (ns)       StdDev (ns)              Name         
 --------  ---------------  ---------  ----------------  ----------------  --------------  --------------  ----------------  ----------------------
     31.4   52,661,820,992          1  52,661,820,992.0  52,661,820,992.0  52,661,820,992  52,661,820,992               0.0  pthread_join          
     30.7   51,427,453,056          9   5,714,161,450.7         325,952.0          15,296  51,260,568,544  17,079,988,325.5  epoll_wait            
     29.0   48,528,988,832        496      97,840,703.3     100,105,616.0           1,792     100,128,192      14,282,594.5  poll                  
      6.2   10,332,804,576         15     688,853,638.4          15,360.0           1,056  10,000,134,432   2,577,249,117.4  futex                 
      1.5    2,518,010,176    113,371          22,210.4          16,384.0           1,024      15,371,232          56,086.3  ioctl                 
      0.8    1,347,542,368          6     224,590,394.7     133,490,496.0           1,888     766,924,512     290,305,957.1  flock                 
      0.3      584,334,656      3,276         178,368.3         204,128.0           1,376       6,626,816         201,768.1  open64                
      0.0       16,626,560      6,640           2,504.0           1,536.0           1,088         304,832          13,230.4  munmap                
      0.0       11,865,344      2,579           4,600.8           3,296.0           1,024         297,248           8,367.4  read                  
      0.0        5,791,552         43         134,687.3           3,584.0           1,632       4,974,240         760,924.8  fopen                 
      0.0        4,804,928      1,883           2,551.7           1,664.0           1,024          25,984           2,370.6  mmap64                
      0.0        4,010,528        178          22,531.1          26,320.0           1,024          48,832           9,603.6  pthread_rwlock_wrlock 
      0.0        3,978,176          1       3,978,176.0       3,978,176.0       3,978,176       3,978,176               0.0  posix_spawnp          
      0.0        3,677,952        159          23,131.8          22,176.0           1,024         102,400          15,647.4  pthread_rwlock_rdlock 
      0.0        1,168,704          2         584,352.0         584,352.0           2,080       1,166,624         823,457.0  pthread_mutex_trylock 
      0.0          913,504        149           6,130.9           5,504.0           1,216          35,936           2,821.1  pthread_cond_signal   
      0.0          488,416         56           8,721.7           6,496.0           1,120          22,912           5,680.7  mmap                  
      0.0          423,712          8          52,964.0           3,968.0           1,664         260,960          96,361.5  fopen64               
      0.0          403,424          4         100,856.0          96,288.0          90,208         120,640          13,510.5  pthread_create        
      0.0          149,600          2          74,800.0          74,800.0          31,552         118,048          61,161.9  waitpid               
      0.0          131,424          8          16,428.0          15,280.0          10,080          23,424           5,902.8  sem_timedwait         
      0.0          103,296          5          20,659.2          22,976.0          11,744          27,968           6,276.8  getline               
      0.0           64,480          1          64,480.0          64,480.0          64,480          64,480               0.0  pthread_cond_wait     
      0.0           52,800          5          10,560.0           4,960.0           1,216          34,912          14,049.0  fgets                 
      0.0           50,784         23           2,208.0           1,792.0           1,024           6,912           1,359.0  fclose                
      0.0           48,992         20           2,449.6           2,000.0           1,344           9,152           1,674.5  write                 
      0.0           40,320         10           4,032.0           3,856.0           1,280           7,456           2,276.4  open                  
      0.0           24,064          5           4,812.8           5,184.0           2,144           7,680           2,209.0  pipe2                 
      0.0           21,312          4           5,328.0           5,680.0           3,648           6,304           1,184.3  fread                 
      0.0           12,736          2           6,368.0           6,368.0           3,936           8,800           3,439.4  socket                
      0.0           10,144          4           2,536.0           2,640.0           1,696           3,168             749.3  epoll_ctl             
      0.0            8,352          2           4,176.0           4,176.0           1,152           7,200           4,276.6  pthread_mutex_lock    
      0.0            8,320          1           8,320.0           8,320.0           8,320           8,320               0.0  socketpair            
      0.0            7,936          1           7,936.0           7,936.0           7,936           7,936               0.0  connect               
      0.0            7,808          4           1,952.0           1,600.0           1,376           3,232             861.4  mprotect              
      0.0            4,768          2           2,384.0           2,384.0           2,016           2,752             520.4  fwrite                
      0.0            3,328          3           1,109.3           1,120.0           1,056           1,152              48.9  fcntl                 
      0.0            2,880          2           1,440.0           1,440.0           1,024           1,856             588.3  signal                
      0.0            2,784          2           1,392.0           1,392.0           1,152           1,632             339.4  sigaction             
      0.0            2,528          1           2,528.0           2,528.0           2,528           2,528               0.0  pthread_cond_broadcast
      0.0            1,376          1           1,376.0           1,376.0           1,376           1,376               0.0  bind                  
      0.0            1,248          1           1,248.0           1,248.0           1,248           1,248               0.0  listen                

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)   Min (ns)   Max (ns)   StdDev (ns)                        Name                      
 --------  ---------------  ---------  ---------  ---------  --------  ----------  -----------  ------------------------------------------------
     35.3    6,510,243,744    749,593    8,685.0    6,208.0     1,376  99,736,352    116,208.7  cudaMalloc                                      
     27.6    5,083,280,288    749,599    6,781.3    4,992.0       576   1,615,776     11,571.5  cudaFree                                        
     19.0    3,507,584,192    625,448    5,608.1    5,696.0     1,920     674,464      1,962.5  cuLaunchKernel                                  
      8.6    1,586,022,112    285,329    5,558.6    4,768.0     1,888   3,466,496      7,755.2  cudaLaunchKernel                                
      6.5    1,208,254,784     93,651   12,901.7   13,248.0        64   3,241,920     10,898.3  cudaMemcpyAsync                                 
      0.9      165,964,768     87,252    1,902.1    1,184.0       480      24,544      2,065.0  cudaStreamSynchronize                           
      0.5       95,354,496     15,626    6,102.3    6,976.0     2,688      28,384      2,556.4  cudaLaunchKernelExC_v11060                      
      0.5       92,008,928    162,414      566.5      544.0       192      22,816        237.7  cudaStreamIsCapturing_v10000                    
      0.2       42,420,032     84,538      501.8      416.0       160     180,384        656.7  cuKernelGetFunction                             
      0.2       40,630,368      1,983   20,489.3    3,456.0     1,984     897,600     50,582.0  cudaStreamCreateWithFlags                       
      0.2       37,229,824         81  459,627.5   97,312.0    82,208   1,814,048    612,156.6  cuModuleLoadData                                
      0.1       26,602,272         48  554,214.0   87,568.0    48,864   1,469,088    543,916.1  cuModuleUnload                                  
      0.1       18,043,328      7,833    2,303.5    2,272.0       640      12,192        403.5  cudaEventQuery                                  
      0.1       13,138,720      7,775    1,689.9    1,664.0     1,344       9,344        338.5  cudaEventRecord                                 
      0.0        7,551,072      1,544    4,890.6    2,624.0     1,952   1,420,288     51,037.8  cudaLaunchKernel_ptsz                           
      0.0        5,962,368     46,283      128.8       96.0        32      20,224        143.0  cuStreamGetCaptureInfo_v2                       
      0.0        4,787,072      1,088    4,399.9    3,680.0     1,664     453,600     15,366.6  cudaStreamDestroy                               
      0.0        1,445,856         15   96,390.4  101,632.0    40,000     168,448     40,346.6  cuLibraryLoadData                               
      0.0          818,912      4,620      177.3      128.0        32       6,240        157.5  cuGetProcAddress_v2                             
      0.0          548,064          8   68,508.0   13,168.0     4,064     474,720    164,198.4  cudaHostAlloc                                   
      0.0          444,160         20   22,208.0   11,824.0     7,008     127,872     26,819.1  cudaDeviceSynchronize                           
      0.0          440,416         46    9,574.3    9,024.0     4,704      26,464      4,208.3  cudaMemsetAsync                                 
      0.0          304,064        128    2,375.5    2,288.0     1,184       8,512        990.8  cudaMemsetAsync_ptsz                            
      0.0          299,904          1  299,904.0  299,904.0   299,904     299,904          0.0  cuLibraryUnload                                 
      0.0          144,256        152      949.1      736.0       160       3,968        728.6  cudaEventDestroy                                
      0.0          137,664        150      917.8      448.0       192       3,296        852.7  cudaEventCreateWithFlags                        
      0.0           32,320          1   32,320.0   32,320.0    32,320      32,320          0.0  cudaMemGetInfo                                  
      0.0           19,936         13    1,533.5    1,536.0     1,344       1,728        135.4  cuInit                                          
      0.0           13,248         35      378.5      288.0       224       1,664        288.9  cudaOccupancyMaxActiveClusters_v11070           
      0.0           12,832         16      802.0      544.0       480       2,496        612.2  cuLibraryGetKernel                              
      0.0           10,144          1   10,144.0   10,144.0    10,144      10,144          0.0  cudaStreamCreate                                
      0.0            5,920          1    5,920.0    5,920.0     5,920       5,920          0.0  cudaMemcpyAsync_ptsz                            
      0.0            4,032         12      336.0      128.0        96       2,784        771.1  cuModuleGetLoadingMode                          
      0.0            3,776          3    1,258.7    1,056.0       768       1,952        617.5  cudaEventCreate                                 
      0.0            3,200          2    1,600.0    1,600.0     1,344       1,856        362.0  cudaOccupancyAvailableDynamicSMemPerBlock_v10200
      0.0            1,408          2      704.0      704.0       448         960        362.0  cudaGetDriverEntryPoint_v11030                  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     19.8      533,418,046     30,622   17,419.4   17,408.0    16,448    18,880        327.6  void execute_dist<interval_32_functor, curand_pseudo_state<curandStateXORWOW>, int, unsigned int, u…
     10.4      281,437,056      7,708   36,512.3   36,768.0    13,664    37,504      2,055.3  sm90_xmma_gemm_f64f64_f64f64_f64_tt_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      9.8      262,841,181     15,312   17,165.7   17,120.0    16,288    18,592        304.8  void execute_dist<random_uniform_functor, curand_pseudo_state<curandStateXORWOW>, double, >(long, l…
      5.1      136,542,017     53,706    2,542.4    2,528.0     2,080     3,488        240.6  void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_nn_align1>(T1::Params)                
      4.9      131,789,600      7,708   17,097.8   17,056.0    16,288    18,464        294.9  void execute_dist<standard_normal_functor, curand_pseudo_state<curandStateXORWOW>, double, >(long, …
      3.6       95,718,561      7,708   12,418.1   12,416.0    12,192    12,832         98.8  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceRadixSortS…
      3.3       88,740,864     15,416    5,756.4    5,760.0     5,472     6,528        146.9  void geqr2_smem<double, double, (int)8, (int)5, (int)5>(int, int, T1 *, unsigned long, T1 *, int)   
      2.7       71,665,310     30,832    2,324.4    2,368.0     2,048     2,976        152.8  void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_tn_align1>(T1::Params)                
      2.5       67,416,864     54,069    1,246.9    1,216.0       896     8,608        219.6  cupy_copy__float64_float64                                                                          
      2.1       56,554,655     30,767    1,838.2    1,888.0     1,504    10,560        185.5  cupy_take                                                                                           
      2.0       53,618,080     38,330    1,398.9    1,312.0       992     2,048        167.9  cupy_multiply__float64_float64_float64                                                              
      2.0       52,703,136     30,972    1,701.6    1,728.0     1,408     2,912        143.7  cupy_scan_naive                                                                                     
      1.9       50,762,976     46,248    1,097.6    1,120.0       736     1,760        157.0  copy_info_kernel(int, int *)                                                                        
      1.7       46,869,921     23,224    2,018.2    2,016.0     1,696     2,752        161.3  cupy_mean                                                                                           
      1.6       43,384,544     23,124    1,876.2    1,888.0     1,536     2,496        144.5  void larft_T_32<double, (int)256>(int, T1 *, int, long, const T1 *, long)                           
      1.4       37,730,979     23,324    1,617.7    1,632.0     1,312     2,368        138.9  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      1.4       36,775,360     30,622    1,200.9    1,184.0       960     1,728        100.6  cupy_add__int_uint32_int64                                                                          
      1.4       36,460,288      7,708    4,730.2    4,736.0     4,576     5,248         70.0  sm90_xmma_gemm_f64f64_f64f64_f64_tt_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      1.3       35,232,704     23,124    1,523.6    1,536.0     1,280     2,048        103.4  void larft_vtv_32<double, (int)128>(long, long, const T1 *, long, long, T1 *, long, long)           
      1.2       33,131,009     23,224    1,426.6    1,440.0     1,184     2,080        105.3  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      1.1       30,675,648     23,124    1,326.6    1,280.0       960     1,824        177.3  cupy_copy__bool_bool                                                                                
      1.0       25,719,552     15,430    1,666.9    1,696.0     1,376     2,816        144.1  cupy_getitem_mask                                                                                   
      0.9       24,971,008     15,576    1,603.2    1,600.0     1,056     5,504        225.7  cupy_subtract__float64_float64_float64                                                              
      0.9       23,619,776     15,516    1,522.3    1,504.0     1,088     2,144        181.0  cupy_true_divide__float64_float_float64                                                             
      0.8       22,687,136     15,516    1,462.2    1,504.0     1,152     2,336        150.8  cupy_true_divide__float64_float64_float64                                                           
      0.8       22,365,248     15,416    1,450.8    1,472.0     1,152     2,048        154.4  cupy_triu_kernel                                                                                    
      0.8       20,261,568     15,416    1,314.3    1,312.0     1,088     1,920        100.6  cupy_cub_sum_pass2                                                                                  
      0.8       20,240,320     15,311    1,321.9    1,312.0     1,088     1,888        102.8  cupy_remainder__int64_int_int64                                                                     
      0.7       19,851,007     15,416    1,287.7    1,280.0     1,056     1,696         98.5  cupy_cub_sum_pass1                                                                                  
      0.7       19,228,032     15,416    1,247.3    1,216.0     1,024     1,664         98.5  void lacpy_kernel<double, double, (int)5, (int)3>(long, long, const T1 *, long, long, T2 *, long, l…
      0.7       18,530,144     15,311    1,210.3    1,216.0       992     1,792        102.8  cupy_add__int64_int64_int64                                                                         
      0.7       18,523,296      7,708    2,403.1    2,400.0     2,016     2,784        157.3  std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (b…
      0.7       18,420,448     15,311    1,203.1    1,184.0       992     1,504         99.5  cupy_subtract__float_float64_float64                                                                
      0.7       17,930,816     15,416    1,163.1    1,152.0       928     1,536         99.6  cupy_equal__int64_int_bool                                                                          
      0.7       17,831,424     15,386    1,158.9    1,152.0       928     4,000        111.6  cupy_add__float64_float64_float64                                                                   
      0.7       17,522,976     15,561    1,126.1    1,120.0       896     1,952        100.2  cupy_copy__int64_int64                                                                              
      0.6       17,343,264     15,516    1,117.8    1,120.0       896     1,440         98.9  cupy_copy__int32_int32                                                                              
      0.6       16,467,008     15,416    1,068.2    1,056.0       864     1,568         99.4  void batch_eye_kernel<double, (int)5, (int)3>(long, long, T1 *, unsigned long, long long, int)      
      0.5       14,745,216     15,416      956.5      960.0       736     1,312        101.9  void orgqr_step1_kernel<double, (int)5, (int)3>(int, int, int, T1 *, int)                           
      0.5       14,128,992      7,868    1,795.8    1,824.0     1,504     4,064        162.9  cupy_power__float64_float_float64                                                                   
      0.5       13,081,344      7,838    1,669.0    1,536.0     1,376   128,832      3,178.6  _norm_ord2                                                                                          
      0.4       11,071,169      7,708    1,436.3    1,472.0     1,152     1,856        144.5  cupy_sqrt__float64_float64                                                                          
      0.4       10,801,919      7,708    1,401.4    1,376.0     1,184     1,792         98.9  cupy_cub_mean_pass2                                                                                 
      0.4       10,667,904      7,883    1,353.3    1,344.0       928     2,784        179.2  cupy_multiply__float_float64_float64                                                                
      0.4       10,279,776      7,708    1,333.7    1,312.0     1,120     1,632         97.7  cupy_cub_mean_pass1                                                                                 
      0.4       10,220,096      7,708    1,325.9    1,280.0     1,120     1,632         97.8  _exists_nan_pass1                                                                                   
      0.4       10,161,216      7,708    1,318.3    1,312.0     1,120     1,728        101.6  _exists_nan_pass2                                                                                   
      0.4       10,082,143      7,708    1,308.0    1,248.0       992     1,856        113.2  void larfb_set_v1_R_kernel<double, (int)16>(long, T1 *, long, long, T1 *, long, long)               
      0.4        9,631,616      7,708    1,249.6    1,280.0     1,056     1,696        102.6  void larfb_restore_v1_R_kernel<double, (int)16>(long, T1 *, long, long, T1 *, long, long)           
      0.3        9,316,448      7,708    1,208.7    1,216.0       992     1,504         97.6  cupy_where__bool_float_float64_float64                                                              
      0.3        9,168,512      7,808    1,174.2    1,184.0       960     1,664        109.4  cupy_greater__float64_float_bool                                                                    
      0.3        9,093,633      7,723    1,177.5    1,120.0       960     2,880        101.3  cupy_greater__float64_float64_bool                                                                  
      0.3        9,077,440      7,708    1,177.7    1,184.0       960     1,792         98.1  cupy_less_equal__float64_float64_bool                                                               
      0.0        1,151,392          2  575,696.0  575,696.0   571,712   579,680      5,634.2  void init_curand<curand_pseudo_state<curandStateXORWOW>>(long, unsigned long, long)                 
      0.0          683,168         42   16,265.9   15,360.0    11,744    92,192     12,111.0  void cutlass_raft_cutlass::Kernel<cuvs::gemm::kernel::FusedDistanceNNPersistent<cutlass_raft_cutlas…
      0.0          627,904         99    6,342.5    6,304.0     6,208     6,688        101.2  void cutlass_raft_cutlass::Kernel2<cutlass_raft_cutlass::gemm::kernel::GemmWithFusedEpilogue<cutlas…
      0.0          616,544         20   30,827.2   27,536.0     3,648    49,600     18,122.2  std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (b…
      0.0          537,056        150    3,580.4    3,552.0     3,136     4,160        231.8  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      0.0          397,184         10   39,718.4   36,432.0    10,080    79,104     26,530.6  void cutlass_raft_cutlass::Kernel<cuvs::gemm::kernel::FusedDistanceNNPersistent<cutlass_raft_cutlas…
      0.0          378,464        255    1,484.2    1,504.0     1,216     1,888        112.8  void raft::linalg::detail::coalescedSumThinKernel<raft::linalg::detail::ReductionThinPolicy<(int)4,…
      0.0          343,392         99    3,468.6    3,360.0     3,296     3,648        143.1  void raft::random::detail::sample_with_replacement_kernel<raft::random::detail::PCGenerator, int, d…
      0.0          240,608         99    2,430.4    2,496.0     2,304     2,560         94.0  void raft::linalg::detail::coalescedSumMediumKernel<(int)256, double, double, int, raft::identity_o…
      0.0          239,232         99    2,416.5    2,464.0     2,272     2,528         80.0  void cub::detail::scan::DeviceScanKernel<cub::detail::scan::policy_hub<double, double, double, unsi…
      0.0          234,080        107    2,187.7    2,176.0     1,856     2,304         69.0  cupy_nonzero_kernel_incomplete_scan                                                                 
      0.0          213,600        119    1,795.0    1,792.0     1,376     2,176         98.8  cupy_bsum_shfl                                                                                      
      0.0          206,560         99    2,086.5    2,080.0     2,016     2,144         42.6  void raft::matrix::detail::matrixLinewiseVecRowsMainKernel<double, int, (unsigned long)16, (int)256…
      0.0          197,888         60    3,298.1    3,328.0     2,752     3,872        278.2  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_k…
      0.0          145,504         99    1,469.7    1,536.0     1,344     1,600         97.8  void raft::matrix::detail::matrixLinewiseVecRowsTailKernel<double, int, (unsigned long)32, raft::mi…
      0.0          143,136         99    1,445.8    1,376.0     1,312     1,568         97.6  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<cub::KeyValu…
      0.0          138,272         99    1,396.7    1,376.0     1,248     1,536         93.5  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0          137,696        100    1,377.0    1,376.0     1,344     1,568         27.5  cupy_equal__int32_int_bool                                                                          
      0.0          126,304         40    3,157.6    3,136.0     2,816     3,648        182.8  void gemmSN_NN_kernel<double, (int)128, (int)2, (int)4, (int)8, (int)3, (int)4, (bool)0, cublasGemv…
      0.0          112,704         38    2,965.9    3,616.0     1,408     3,744      1,037.0  void raft::linalg::detail::sum_rows_by_key_large_nkeys_kernel_rowmajor<double *, thrust::transform_…
      0.0          112,288         99    1,134.2    1,216.0       928     1,376        146.6  void cub::detail::scan::DeviceScanInitKernel<cub::ScanTileState<double, (bool)1>>(T1, int)          
      0.0          108,096         90    1,201.1    1,136.0       800     2,240        334.5  cupy_fill                                                                                           
      0.0           77,696         52    1,494.2    1,504.0     1,152     1,632        119.2  void cuvs::distance::detail::initBinMutexKernel<int>(cuda::__4::counting_semaphore<(cuda::std::__4:…
      0.0           65,984         24    2,749.3    2,448.0     1,536     5,856      1,087.2  cupy_concatenate_same_size                                                                          
      0.0           60,416         38    1,589.9    1,584.0     1,472     1,728         96.7  void raft::matrix::detail::matrixLinewiseVecColsMainKernel<double, int, (unsigned long)16, (int)256…
      0.0           59,264         38    1,559.6    1,568.0     1,408     1,728        144.0  void raft::linalg::detail::mapThenReduceKernel<double, double, unsigned long, raft::compose_op<raft…
      0.0           56,832         42    1,353.1    1,408.0     1,120     1,440         93.0  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, long, …
      0.0           55,008         26    2,115.7    2,112.0     2,080     2,176         31.8  void raft::linalg::detail::reduce_cols_by_key_cached_kernel<double, thrust::transform_iterator<cuvs…
      0.0           45,888         29    1,582.3    1,536.0     1,216     1,984        220.5  cupy_cub_any_pass1                                                                                  
      0.0           44,576         38    1,173.1    1,120.0     1,056     1,312         96.2  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0           44,096         30    1,469.9    1,344.0     1,088     2,624        374.0  cupy_negative__float64_float64                                                                      
      0.0           43,488         29    1,499.6    1,504.0     1,248     2,016        211.9  cupy_cub_any_pass2                                                                                  
      0.0           39,584          8    4,948.0    4,960.0     4,832     5,056         78.3  void raft::random::detail::rngKernel<(int)1, double, int, raft::random::detail::PCGenerator, raft::…
      0.0           35,616         15    2,374.4    2,304.0     2,016     3,008        321.0  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceScanKernel…
      0.0           29,088          8    3,636.0    3,664.0     3,328     3,776        139.9  void cub::detail::select::DeviceSelectSweepKernel<cub::detail::select::policy_hub<cub::KeyValuePair…
      0.0           26,304         14    1,878.9    1,472.0     1,120     4,800        975.0  cupy_bincount_kernel                                                                                
      0.0           25,536         15    1,702.4    1,600.0     1,120     2,784        466.0  cupy_maximum__float_float64_float64                                                                 
      0.0           20,192         14    1,442.3    1,456.0     1,152     1,664        142.8  cupy_less__int32_int_bool                                                                           
      0.0           19,936          9    2,215.1    2,272.0     1,984     2,336        118.4  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0           19,264         16    1,204.0    1,216.0       928     1,344        117.4  cupy_arange__int_int_int64                                                                          
      0.0           18,752         14    1,339.4    1,312.0     1,152     1,600        149.5  cupy_greater__int64_int_bool                                                                        
      0.0           17,472         12    1,456.0    1,440.0     1,344     1,568         91.0  void raft::linalg::detail::reduce_cols_by_key_direct_kernel<double, thrust::transform_iterator<cuvs…
      0.0           17,216         15    1,147.7    1,056.0       960     1,344        165.8  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceScanInitKe…
      0.0           16,736         15    1,115.7    1,088.0       896     1,312        100.4  cupy_arange__int_int_int32                                                                          
      0.0           16,704          7    2,386.3    2,368.0     2,112     2,592        153.3  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, double, double, double, double, (bool)0, …
      0.0           16,096         11    1,463.3    1,536.0     1,312     1,536         96.1  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0           15,552          8    1,944.0    1,920.0     1,632     2,368        214.2  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceKern…
      0.0           12,320          7    1,760.0    1,792.0     1,376     1,984        192.9  cupy_nonzero_kernel                                                                                 
      0.0           11,872         10    1,187.2    1,184.0       928     1,312        107.1  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, long, …
      0.0           11,680          8    1,460.0    1,456.0     1,344     1,600        109.4  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0           11,648          8    1,456.0    1,472.0     1,312     1,536         88.9  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.0           11,552          1   11,552.0   11,552.0    11,552    11,552          0.0  void cub::detail::histogram::DeviceHistogramSweepKernel<cub::detail::histogram::policy_hub<int, dou…
      0.0           10,784          7    1,540.6    1,504.0     1,248     1,952        227.3  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.0           10,016          8    1,252.0    1,264.0     1,120     1,344         87.9  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, int, c…
      0.0            9,472          8    1,184.0    1,184.0     1,024     1,376        155.8  void cub::detail::scan::DeviceCompactInitKernel<cub::ScanTileState<int, (bool)1>, int *>(T1, int, T…
      0.0            9,344          2    4,672.0    4,672.0     4,192     5,152        678.8  cupy_concatenate                                                                                    
      0.0            7,040          3    2,346.7    2,240.0     2,240     2,560        184.8  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (…
      0.0            6,144          1    6,144.0    6,144.0     6,144     6,144          0.0  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (…
      0.0            4,416          2    2,208.0    2,208.0     2,112     2,304        135.8  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0            4,128          2    2,064.0    2,064.0     2,016     2,112         67.9  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0            3,616          2    1,808.0    1,808.0     1,632     1,984        248.9  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0            3,040          2    1,520.0    1,520.0     1,504     1,536         22.6  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            2,304          2    1,152.0    1,152.0       960     1,344        271.5  cupy_copy__int_float64                                                                              
      0.0            2,016          1    2,016.0    2,016.0     2,016     2,016          0.0  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0            1,568          1    1,568.0    1,568.0     1,568     1,568          0.0  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            1,440          1    1,440.0    1,440.0     1,440     1,440          0.0  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            1,280          1    1,280.0    1,280.0     1,280     1,280          0.0  cupy_copy__int_int32                                                                                
      0.0            1,056          1    1,056.0    1,056.0     1,056     1,056          0.0  void cub::detail::histogram::DeviceHistogramInitKernel<(int)1, double, int>(cuda::std::__4::array<i…

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  ------  --------  --------  --------  --------  -----------  ------------------------------
     80.8      107,212,192  70,002   1,531.6   1,408.0     1,312     3,392        244.7  [CUDA memcpy Device-to-Host]  
     18.9       25,120,576  23,388   1,074.1   1,088.0       832     1,664         99.9  [CUDA memcpy Device-to-Device]
      0.2          205,920     174   1,183.4   1,168.0       736     5,024        437.1  [CUDA memset]                 
      0.1          158,528     147   1,078.4     896.0       704     3,616        513.2  [CUDA memcpy Host-to-Device]  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  ------  --------  --------  --------  --------  -----------  ------------------------------
     34.304     174     0.197     0.002     0.000    12.976        1.396  [CUDA memset]                 
      6.933  23,388     0.000     0.000     0.000     0.080        0.001  [CUDA memcpy Device-to-Device]
      0.396  70,002     0.000     0.000     0.000     0.016        0.000  [CUDA memcpy Device-to-Host]  
      0.390     147     0.003     0.000     0.000     0.033        0.007  [CUDA memcpy Host-to-Device]  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/openmp_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain OpenMP event data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/opengl_khr_range_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/opengl_khr_gpu_range_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/vulkan_marker_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/vulkan_gpu_marker_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/dx11_pix_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain DX11 CPU debug markers.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/dx12_gpu_marker_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain DX12 GPU debug markers.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/dx12_pix_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain DX12 CPU debug markers.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/wddm_queue_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain WDDM context data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/um_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/um_total_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/um_cpu_page_faults_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/openacc_sum.py]... 
SKIPPED: gpu_adaptive_clustering_profile.sqlite does not contain OpenACC event data.
```