## Generation of Nsight Systems Profile for GPU Adaptive Clustering
```sh
c608-082[gh](653)$ nsys profile --trace=cuda,nvtx,osrt --output=gpu_adaptive_clustering_profile --force-overwrite true uv run python src/gpu_adaptive_clustering_optimized.py
ABS_TOL=0.001, REL_TOL=0.001
Running on GPU. N_SAMPLES=10,000
  --> 100/100 clusters need splitting
Level 1: processing 200 clusters (prev level took 0.54s)
  --> 200/200 clusters need splitting
Level 2: processing 400 clusters (prev level took 0.84s)
  --> 400/400 clusters need splitting
Level 3: processing 800 clusters (prev level took 1.79s)
  --> 799/800 clusters need splitting
Level 4: processing 1598 clusters (prev level took 3.56s)
  --> 1504/1598 clusters need splitting
Level 5: processing 3008 clusters (prev level took 6.96s)
  --> 2061/3008 clusters need splitting
Level 6: processing 4122 clusters (prev level took 10.63s)
  --> 1568/4122 clusters need splitting
Level 7: processing 3136 clusters (prev level took 10.61s)
  --> 694/3136 clusters need splitting
Level 8: processing 1388 clusters (prev level took 6.36s)
  --> 274/1388 clusters need splitting
Level 9: processing 548 clusters (prev level took 2.73s)
  --> 81/548 clusters need splitting
Level 10: processing 162 clusters (prev level took 1.01s)
  --> 20/162 clusters need splitting
Level 11: processing 40 clusters (prev level took 0.30s)
  --> 5/40 clusters need splitting
Level 12: processing 10 clusters (prev level took 0.08s)
  --> 1/10 clusters need splitting
Level 13: processing 2 clusters (prev level took 0.03s)
  --> 1/2 clusters need splitting
Level 14: processing 2 clusters (prev level took 0.01s)
  --> 0/2 clusters need splitting
Total final clusters formed: 7808
Adaptive clustering completed in 45.74 seconds.
Generating '/tmp/nsys-report-3b2e.qdstrm'
[1/1] [========================100%] gpu_adaptive_clustering_profile.nsys-rep
Generated:
    /home1/10973/prajwal_prathiksh/github-repos/knnsur/gpu_adaptive_clustering_profile.nsys-rep
```

## Analysis of Nsight Systems Profile
```sh
c608-041[gh](660)$ nsys stats gpu_adaptive_clustering_profile.nsys-rep
Generating SQLite file gpu_adaptive_clustering_profile.sqlite from gpu_adaptive_clustering_profile.nsys-rep
Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)  Min (ns)  Max (ns)   StdDev (ns)   Style                       Range                    
 --------  ---------------  ---------  ---------  --------  --------  ---------  -----------  -------  ---------------------------------------------
     45.2      286,363,008     23,336   12,271.3  12,160.0     7,808     44,384      1,551.1  PushPop  CCCL:cub::DeviceReduce::Reduce               
     41.0      260,088,544     23,226   11,198.2  11,040.0     7,936     46,656      1,400.4  PushPop  CCCL:cub::DeviceReduce::Sum                  
     13.2       83,561,472      7,708   10,840.9  10,688.0     8,352     45,664      1,341.6  PushPop  CCCL:cub::DeviceRadixSort                    
      0.3        1,596,288         15  106,419.2  23,072.0    11,776  1,281,632    325,224.3  PushPop  CCCL:cub::DeviceReduce::Max                  
      0.2        1,312,352        114   11,511.9   8,656.0     7,552     44,672      7,466.0  PushPop  CCCL:cub::DeviceScan::InclusiveSum           
      0.1          417,024         99    4,212.4   3,936.0     3,360     24,448      2,125.7  PushPop  CCCL:cub::DeviceReduce::ArgMin               
      0.0          250,976         52    4,826.5   4,000.0     3,392     19,296      2,898.7  PushPop  CCCL:cub::DeviceFor::Bulk                    
      0.0           96,032          8   12,004.0   8,352.0     8,064     37,280     10,217.3  PushPop  CCCL:cub::DeviceSelect::If                   
      0.0           50,080          8    6,260.0   5,088.0     4,672     15,104      3,581.3  PushPop  CCCL:cub::DeviceFor::ForEachN                
      0.0           24,416          1   24,416.0  24,416.0    24,416     24,416          0.0  PushPop  CCCL:cub::DeviceHistogram::MultiHistogramEven

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls      Avg (ns)          Med (ns)         Min (ns)        Max (ns)       StdDev (ns)              Name         
 --------  ---------------  ---------  ----------------  ----------------  --------------  --------------  ----------------  ----------------------
     31.2   49,249,528,768          1  49,249,528,768.0  49,249,528,768.0  49,249,528,768  49,249,528,768               0.0  pthread_join          
     30.6   48,211,892,256          9   5,356,876,917.3         242,432.0          16,288  48,204,501,888  16,067,859,466.6  epoll_wait            
     29.2   46,000,346,496        471      97,665,279.2     100,108,352.0           1,440     100,113,440      14,770,979.0  poll                  
      6.4   10,045,419,840         15     669,694,656.0          11,520.0           1,056  10,000,129,664   2,581,205,215.1  futex                 
      1.6    2,533,294,560    111,593          22,701.2          16,672.0           1,024      15,323,648          61,405.1  ioctl                 
      0.6    1,009,223,776          6     168,203,962.7      17,059,712.0           1,664     619,889,440     261,350,063.5  flock                 
      0.4      606,230,208      3,271         185,334.8         216,256.0           1,152       3,391,328         154,549.9  open64                
      0.0       13,577,312      6,585           2,061.9           1,632.0           1,248         292,576           5,924.5  munmap                
      0.0       11,867,776      2,571           4,616.0           3,328.0           1,024         251,200           7,839.6  read                  
      0.0        4,870,496      1,981           2,458.6           1,568.0           1,024          30,848           2,356.0  mmap64                
      0.0        3,784,128        161          23,503.9          26,944.0           1,088          39,968           9,190.6  pthread_rwlock_wrlock 
      0.0        3,722,464        162          22,978.2          23,952.0           1,088         108,352          12,222.5  pthread_rwlock_rdlock 
      0.0          497,504         81           6,142.0           5,568.0           1,184          22,112           2,382.0  pthread_cond_signal   
      0.0          485,504         43          11,290.8           3,392.0           1,536         238,848          35,896.6  fopen                 
      0.0          469,376          1         469,376.0         469,376.0         469,376         469,376               0.0  posix_spawnp          
      0.0          451,776          8          56,472.0           3,264.0           1,856         297,760         107,150.1  fopen64               
      0.0          404,448         46           8,792.3           9,920.0           1,120          21,440           5,615.9  mmap                  
      0.0          389,888          4          97,472.0          93,520.0          89,856         112,992          10,493.4  pthread_create        
      0.0          213,952          9          23,772.4          14,112.0          10,400          94,336          26,844.3  sem_timedwait         
      0.0          148,032          2          74,016.0          74,016.0          30,944         117,088          60,913.0  waitpid               
      0.0          107,200          5          21,440.0          23,136.0          11,328          30,304           7,182.9  getline               
      0.0           69,920          1          69,920.0          69,920.0          69,920          69,920               0.0  pthread_cond_wait     
      0.0           59,680          4          14,920.0           8,912.0           1,760          40,096          17,356.7  fgets                 
      0.0           52,864         21           2,517.3           1,984.0           1,088           9,728           1,773.6  write                 
      0.0           49,216         20           2,460.8           2,000.0           1,024           6,752           1,448.1  fclose                
      0.0           38,592          9           4,288.0           4,800.0           1,856           7,552           2,029.5  open                  
      0.0           27,520          5           5,504.0           5,440.0           1,984           8,960           3,087.0  pipe2                 
      0.0           25,120          4           6,280.0           6,576.0           3,552           8,416           2,030.2  fread                 
      0.0           13,888          3           4,629.3           1,568.0           1,120          11,200           5,694.8  pthread_mutex_lock    
      0.0           12,192          2           6,096.0           6,096.0           4,032           8,160           2,918.9  socket                
      0.0           10,944          4           2,736.0           3,136.0           1,216           3,456           1,045.6  epoll_ctl             
      0.0            8,320          4           2,080.0           1,760.0           1,504           3,296             829.9  mprotect              
      0.0            8,064          1           8,064.0           8,064.0           8,064           8,064               0.0  socketpair            
      0.0            7,584          1           7,584.0           7,584.0           7,584           7,584               0.0  connect               
      0.0            4,576          2           2,288.0           2,288.0           1,856           2,720             610.9  fwrite                
      0.0            3,296          1           3,296.0           3,296.0           3,296           3,296               0.0  pthread_cond_broadcast
      0.0            2,528          2           1,264.0           1,264.0           1,120           1,408             203.6  sigaction             
      0.0            2,368          2           1,184.0           1,184.0           1,152           1,216              45.3  fcntl                 
      0.0            2,304          1           2,304.0           2,304.0           2,304           2,304               0.0  signal                
      0.0            2,208          1           2,208.0           2,208.0           2,208           2,208               0.0  pthread_mutex_trylock 
      0.0            1,984          1           1,984.0           1,984.0           1,984           1,984               0.0  bind                  
      0.0            1,376          1           1,376.0           1,376.0           1,376           1,376               0.0  dup2                  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)   Min (ns)   Max (ns)   StdDev (ns)                        Name                      
 --------  ---------------  ---------  ---------  ---------  --------  ----------  -----------  ------------------------------------------------
     34.3    6,216,947,744    702,945    8,844.1    6,240.0     1,248  95,077,184    114,510.4  cudaMalloc                                      
     26.9    4,879,594,912    702,951    6,941.6    5,024.0       512   1,131,584     11,775.1  cudaFree                                        
     18.0    3,254,604,096    578,800    5,623.0    5,632.0     1,888     429,216      2,032.5  cuLaunchKernel                                  
      9.1    1,649,120,416    285,329    5,779.7    4,864.0     1,856   3,481,568      8,142.3  cudaLaunchKernel                                
      8.6    1,548,731,456    124,684   12,421.3   12,448.0        32   3,118,976      9,086.5  cudaMemcpyAsync                                 
      1.0      176,784,672    118,285    1,494.6    1,088.0       480      17,952      1,764.5  cudaStreamSynchronize                           
      0.6      106,236,480    193,447      549.2      512.0       160      20,352        256.3  cudaStreamIsCapturing_v10000                    
      0.5       97,533,824     15,626    6,241.8    6,944.0     2,624      24,576      2,678.0  cudaLaunchKernelExC_v11060                      
      0.2       42,757,056     84,538      505.8      416.0       160     178,432        650.9  cuKernelGetFunction                             
      0.2       35,525,728         76  467,443.8   95,248.0    82,784   1,617,248    593,172.5  cuModuleLoadData                                
      0.1       22,831,712         48  475,660.7   93,344.0    49,408   1,096,864    451,632.0  cuModuleUnload                                  
      0.1       17,900,448      7,833    2,285.3    2,240.0       768      12,320        394.9  cudaEventQuery                                  
      0.1       17,452,512      1,983    8,801.1    3,328.0     2,176     860,032     27,352.3  cudaStreamCreateWithFlags                       
      0.1       13,114,336      7,775    1,686.7    1,664.0     1,280      10,464        315.7  cudaEventRecord                                 
      0.1       10,072,960      1,544    6,523.9    2,432.0     1,792   2,842,496     88,059.6  cudaLaunchKernel_ptsz                           
      0.0        8,746,048      1,984    4,408.3    4,032.0     1,888      40,736      3,742.8  cudaStreamDestroy                               
      0.0        5,975,648     46,283      129.1       96.0        32       9,536        106.4  cuStreamGetCaptureInfo_v2                       
      0.0        1,430,496         15   95,366.4   99,680.0    40,224     167,072     40,352.3  cuLibraryLoadData                               
      0.0          791,360      4,620      171.3      128.0        32       7,136        184.5  cuGetProcAddress_v2                             
      0.0          535,008          8   66,876.0   10,432.0     3,776     470,624    163,182.9  cudaHostAlloc                                   
      0.0          438,976         46    9,543.0    9,424.0     4,608      26,112      4,033.9  cudaMemsetAsync                                 
      0.0          327,456          1  327,456.0  327,456.0   327,456     327,456          0.0  cuLibraryUnload                                 
      0.0          279,136        128    2,180.8    2,144.0       992       8,000        928.9  cudaMemsetAsync_ptsz                            
      0.0          198,944        150    1,326.3      416.0       192      28,384      3,244.0  cudaEventCreateWithFlags                        
      0.0          141,536        152      931.2      688.0       160       4,352        779.1  cudaEventDestroy                                
      0.0           58,016         20    2,900.8    1,776.0       672      12,480      2,757.6  cudaDeviceSynchronize                           
      0.0           29,952          1   29,952.0   29,952.0    29,952      29,952          0.0  cudaMemGetInfo                                  
      0.0           19,488         13    1,499.1    1,440.0     1,280       2,080        222.4  cuInit                                          
      0.0           13,344         35      381.3      288.0       224       1,536        295.3  cudaOccupancyMaxActiveClusters_v11070           
      0.0           11,872         16      742.0      560.0       480       1,696        426.0  cuLibraryGetKernel                              
      0.0            9,760          1    9,760.0    9,760.0     9,760       9,760          0.0  cudaStreamCreate                                
      0.0            4,672          1    4,672.0    4,672.0     4,672       4,672          0.0  cudaMemcpyAsync_ptsz                            
      0.0            4,256         12      354.7      128.0        64       2,720        746.0  cuModuleGetLoadingMode                          
      0.0            4,096          3    1,365.3    1,120.0       832       2,144        689.5  cudaEventCreate                                 
      0.0            3,680          2    1,840.0    1,840.0     1,728       1,952        158.4  cudaOccupancyAvailableDynamicSMemPerBlock_v10200
      0.0            1,088          2      544.0      544.0       384         704        226.3  cudaGetDriverEntryPoint_v11030                  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     20.4      536,323,165     30,622   17,514.3   17,504.0    16,544    19,104        335.2  void execute_dist<interval_32_functor, curand_pseudo_state<curandStateXORWOW>, int, unsigned int, u…
     10.7      280,959,999      7,708   36,450.4   36,704.0    14,048    37,856      2,050.6  sm90_xmma_gemm_f64f64_f64f64_f64_tt_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
     10.0      263,013,665     15,312   17,177.0   17,152.0    16,320    18,848        308.7  void execute_dist<random_uniform_functor, curand_pseudo_state<curandStateXORWOW>, double, >(long, l…
      5.1      135,309,472     53,706    2,519.4    2,496.0     2,080     3,360        245.0  void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_nn_align1>(T1::Params)                
      5.0      132,075,424      7,708   17,134.9   17,120.0    16,288    18,496        281.6  void execute_dist<standard_normal_functor, curand_pseudo_state<curandStateXORWOW>, double, >(long, …
      3.6       95,036,480      7,708   12,329.6   12,320.0    12,096    12,672        107.7  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceRadixSortS…
      3.4       90,375,489     15,416    5,862.4    5,856.0     5,408     6,592        178.7  void geqr2_smem<double, double, (int)8, (int)5, (int)5>(int, int, T1 *, unsigned long, T1 *, int)   
      2.7       71,309,118     30,832    2,312.8    2,336.0     2,016     2,976        152.7  void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_tn_align1>(T1::Params)                
      2.6       67,254,655     54,069    1,243.9    1,216.0       896     8,800        223.3  cupy_copy__float64_float64                                                                          
      2.1       56,027,648     30,767    1,821.0    1,856.0     1,504    10,080        179.1  cupy_take                                                                                           
      2.0       53,375,136     38,330    1,392.5    1,312.0       992     1,984        160.9  cupy_multiply__float64_float64_float64                                                              
      2.0       52,334,496     30,972    1,689.7    1,696.0     1,408     3,104        143.9  cupy_scan_naive                                                                                     
      1.9       50,166,176     46,248    1,084.7    1,088.0       736     1,728        156.7  copy_info_kernel(int, int *)                                                                        
      1.8       46,575,680     23,224    2,005.5    2,016.0     1,696     2,752        161.3  cupy_mean                                                                                           
      1.6       43,238,496     23,124    1,869.9    1,856.0     1,536     2,688        153.8  void larft_T_32<double, (int)256>(int, T1 *, int, long, const T1 *, long)                           
      1.4       37,386,944     23,324    1,602.9    1,600.0     1,312     2,304        138.6  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      1.4       36,319,648     30,622    1,186.1    1,152.0       960     1,472        100.2  cupy_add__int_uint32_int64                                                                          
      1.4       36,155,585      7,708    4,690.7    4,704.0     4,544     5,312         71.8  sm90_xmma_gemm_f64f64_f64f64_f64_tt_n_tilesize64x64x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      1.4       35,776,576     23,124    1,547.2    1,568.0     1,280     1,984        105.8  void larft_vtv_32<double, (int)128>(long, long, const T1 *, long, long, T1 *, long, long)           
      1.3       33,005,856     23,224    1,421.2    1,408.0     1,184     1,952        110.4  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      1.2       30,963,936     23,124    1,339.0    1,280.0       960     1,824        183.4  cupy_copy__bool_bool                                                                                
      1.0       25,718,560     15,430    1,666.8    1,728.0     1,376     2,784        144.8  cupy_getitem_mask                                                                                   
      0.9       24,598,912     15,576    1,579.3    1,568.0     1,056     5,408        205.5  cupy_subtract__float64_float64_float64                                                              
      0.9       23,040,832     15,416    1,494.6    1,536.0     1,152     1,984        186.8  cupy_triu_kernel                                                                                    
      0.8       21,052,800     15,516    1,356.8    1,376.0     1,088     1,696        145.6  cupy_true_divide__float64_float_float64                                                             
      0.8       20,223,168     15,416    1,311.8    1,312.0     1,088     1,632        100.5  cupy_cub_sum_pass2                                                                                  
      0.8       20,055,584     15,311    1,309.9    1,280.0     1,088     1,728        101.4  cupy_remainder__int64_int_int64                                                                     
      0.8       19,750,848     15,416    1,281.2    1,280.0     1,056     1,600        100.2  cupy_cub_sum_pass1                                                                                  
      0.7       19,263,872     15,416    1,249.6    1,216.0     1,024     1,408         97.8  void lacpy_kernel<double, double, (int)5, (int)3>(long, long, const T1 *, long, long, T2 *, long, l…
      0.7       18,488,416     15,311    1,207.5    1,184.0       992     1,536         98.1  cupy_add__int64_int64_int64                                                                         
      0.7       18,097,119     15,311    1,182.0    1,152.0       960     1,472        100.4  cupy_subtract__float_float64_float64                                                                
      0.7       17,989,056      7,708    2,333.8    2,336.0     1,984     2,752        119.5  std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (b…
      0.7       17,693,120     15,386    1,149.9    1,152.0       928     3,968        112.5  cupy_add__float64_float64_float64                                                                   
      0.7       17,426,305     15,561    1,119.9    1,120.0       896     1,504        100.7  cupy_copy__int64_int64                                                                              
      0.7       17,237,920     15,516    1,111.0    1,088.0       896     1,440        100.5  cupy_copy__int32_int32                                                                              
      0.6       16,466,016     15,416    1,068.1    1,056.0       864     1,248         98.9  void batch_eye_kernel<double, (int)5, (int)3>(long, long, T1 *, unsigned long, long long, int)      
      0.6       15,729,440     15,416    1,020.3    1,024.0       736     1,472        133.0  void orgqr_step1_kernel<double, (int)5, (int)3>(int, int, int, T1 *, int)                           
      0.5       13,028,096      7,838    1,662.2    1,536.0     1,376   128,864      3,179.2  _norm_ord2                                                                                          
      0.4       11,146,400      7,708    1,446.1    1,472.0     1,152     1,824        152.6  cupy_sqrt__float64_float64                                                                          
      0.4       11,008,800      7,708    1,428.2    1,472.0     1,152     1,856        142.0  cupy_true_divide__float64_float64_float64                                                           
      0.4       10,770,528      7,708    1,397.3    1,376.0     1,184     1,824        102.9  cupy_cub_mean_pass2                                                                                 
      0.4       10,248,128      7,708    1,329.5    1,344.0     1,120     1,728        101.9  cupy_cub_mean_pass1                                                                                 
      0.4       10,194,400      7,708    1,322.6    1,344.0     1,120     1,632        101.0  _exists_nan_pass2                                                                                   
      0.4       10,171,168      7,708    1,319.6    1,312.0     1,120     1,824        101.7  _exists_nan_pass1                                                                                   
      0.4       10,082,623      7,708    1,308.1    1,248.0       992     1,760        108.5  void larfb_set_v1_R_kernel<double, (int)16>(long, T1 *, long, long, T1 *, long, long)               
      0.4        9,599,168      7,708    1,245.4    1,248.0     1,056     1,664        103.9  void larfb_restore_v1_R_kernel<double, (int)16>(long, T1 *, long, long, T1 *, long, long)           
      0.4        9,313,344      7,708    1,208.3    1,216.0       992     1,536        100.5  cupy_where__bool_float_float64_float64                                                              
      0.3        9,078,560      7,723    1,175.5    1,184.0       960     3,008        103.8  cupy_greater__float64_float64_bool                                                                  
      0.3        9,032,576      7,708    1,171.8    1,184.0       960     1,504        100.2  cupy_less_equal__float64_float64_bool                                                               
      0.0        1,136,960          2  568,480.0  568,480.0   568,256   568,704        316.8  void init_curand<curand_pseudo_state<curandStateXORWOW>>(long, unsigned long, long)                 
      0.0          669,088         42   15,930.7   15,008.0    11,488    91,616     12,061.1  void cutlass_raft_cutlass::Kernel<cuvs::gemm::kernel::FusedDistanceNNPersistent<cutlass_raft_cutlas…
      0.0          629,376         99    6,357.3    6,336.0     6,144     6,912        115.0  void cutlass_raft_cutlass::Kernel2<cutlass_raft_cutlass::gemm::kernel::GemmWithFusedEpilogue<cutlas…
      0.0          613,600         20   30,680.0   27,376.0     3,616    49,536     17,999.9  std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (b…
      0.0          534,080        150    3,560.5    3,520.0     3,136     4,352        255.3  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage3_warpsize2x2x1_tensor16x8x16_aligna8_a…
      0.0          395,136         10   39,513.6   36,512.0     9,664    79,136     26,270.3  void cutlass_raft_cutlass::Kernel<cuvs::gemm::kernel::FusedDistanceNNPersistent<cutlass_raft_cutlas…
      0.0          377,664        255    1,481.0    1,504.0     1,312     1,760        110.8  void raft::linalg::detail::coalescedSumThinKernel<raft::linalg::detail::ReductionThinPolicy<(int)4,…
      0.0          345,376         99    3,488.6    3,584.0     3,264     3,680        147.3  void raft::random::detail::sample_with_replacement_kernel<raft::random::detail::PCGenerator, int, d…
      0.0          252,128         99    2,546.7    2,464.0     2,400     2,688        101.2  void raft::linalg::detail::coalescedSumMediumKernel<(int)256, double, double, int, raft::identity_o…
      0.0          237,344         99    2,397.4    2,368.0     2,272     2,624         84.6  void cub::detail::scan::DeviceScanKernel<cub::detail::scan::policy_hub<double, double, double, unsi…
      0.0          232,736        107    2,175.1    2,176.0     1,824     2,304         71.6  cupy_nonzero_kernel_incomplete_scan                                                                 
      0.0          212,832        119    1,788.5    1,792.0     1,344     2,144        107.2  cupy_bsum_shfl                                                                                      
      0.0          196,512         60    3,275.2    3,296.0     2,624     3,872        281.5  sm90_xmma_gemm_f64f64_f64f64_f64_nn_n_tilesize32x32x32_stage5_warpsize2x2x1_tensor16x8x16_execute_k…
      0.0          196,352         99    1,983.4    1,984.0     1,888     2,144         47.1  void raft::matrix::detail::matrixLinewiseVecRowsMainKernel<double, int, (unsigned long)16, (int)256…
      0.0          152,448         99    1,539.9    1,472.0     1,408     1,696         96.7  void raft::matrix::detail::matrixLinewiseVecRowsTailKernel<double, int, (unsigned long)32, raft::mi…
      0.0          140,928         99    1,423.5    1,472.0     1,280     1,568         98.7  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<cub::KeyValu…
      0.0          138,144         99    1,395.4    1,472.0     1,248     1,536         97.5  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0          137,856        100    1,378.6    1,376.0     1,344     1,568         26.0  cupy_equal__int32_int_bool                                                                          
      0.0          128,832         60    2,147.2    1,952.0     1,568     4,064        600.3  cupy_power__float64_float_float64                                                                   
      0.0          125,184         40    3,129.6    3,120.0     2,752     3,520        195.4  void gemmSN_NN_kernel<double, (int)128, (int)2, (int)4, (int)8, (int)3, (int)4, (bool)0, cublasGemv…
      0.0          116,448         99    1,176.2    1,088.0       992     1,376        146.1  void cub::detail::scan::DeviceScanInitKernel<cub::ScanTileState<double, (bool)1>>(T1, int)          
      0.0          113,024         38    2,974.3    3,616.0     1,440     3,744      1,017.0  void raft::linalg::detail::sum_rows_by_key_large_nkeys_kernel_rowmajor<double *, thrust::transform_…
      0.0          111,616         75    1,488.2    1,344.0     1,024     2,784        437.9  cupy_multiply__float_float64_float64                                                                
      0.0          107,424         90    1,193.6    1,120.0       736     2,304        340.9  cupy_fill                                                                                           
      0.0           76,992         52    1,480.6    1,504.0     1,184     1,632        108.5  void cuvs::distance::detail::initBinMutexKernel<int>(cuda::__4::counting_semaphore<(cuda::std::__4:…
      0.0           65,088         24    2,712.0    2,448.0     1,664     5,856      1,060.2  cupy_concatenate_same_size                                                                          
      0.0           59,584         38    1,568.0    1,568.0     1,440     1,696        100.4  void raft::matrix::detail::matrixLinewiseVecColsMainKernel<double, int, (unsigned long)16, (int)256…
      0.0           59,488         38    1,565.5    1,552.0     1,376     1,760        149.4  void raft::linalg::detail::mapThenReduceKernel<double, double, unsigned long, raft::compose_op<raft…
      0.0           56,576         42    1,347.0    1,376.0     1,120     1,440         97.4  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, long, …
      0.0           55,840         26    2,147.7    2,144.0     2,080     2,208         36.6  void raft::linalg::detail::reduce_cols_by_key_cached_kernel<double, thrust::transform_iterator<cuvs…
      0.0           45,568         29    1,571.3    1,536.0     1,184     1,920        215.4  cupy_cub_any_pass1                                                                                  
      0.0           43,520         38    1,145.3    1,200.0       992     1,280        102.3  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0           43,328         30    1,444.3    1,344.0       992     2,624        388.2  cupy_negative__float64_float64                                                                      
      0.0           42,944         29    1,480.8    1,440.0     1,152     2,112        232.9  cupy_cub_any_pass2                                                                                  
      0.0           39,424          8    4,928.0    4,960.0     4,800     5,024        102.6  void raft::random::detail::rngKernel<(int)1, double, int, raft::random::detail::PCGenerator, raft::…
      0.0           34,880         15    2,325.3    2,240.0     1,952     2,944        311.1  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceScanKernel…
      0.0           29,248          8    3,656.0    3,680.0     3,456     3,776        102.3  void cub::detail::select::DeviceSelectSweepKernel<cub::detail::select::policy_hub<cub::KeyValuePair…
      0.0           25,504         14    1,821.7    1,456.0     1,216     4,736        959.5  cupy_bincount_kernel                                                                                
      0.0           24,928         15    1,661.9    1,568.0     1,088     2,816        477.2  cupy_maximum__float_float64_float64                                                                 
      0.0           19,904          9    2,211.6    2,144.0     2,080     2,400        131.4  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0           19,680         14    1,405.7    1,408.0     1,152     1,632        134.9  cupy_less__int32_int_bool                                                                           
      0.0           19,168         16    1,198.0    1,216.0       960     1,408        125.3  cupy_arange__int_int_int64                                                                          
      0.0           17,920         14    1,280.0    1,280.0       992     1,536        190.8  cupy_greater__int64_int_bool                                                                        
      0.0           17,312         15    1,154.1    1,056.0       960     1,344        147.9  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceScanInitKe…
      0.0           17,088         12    1,424.0    1,424.0     1,312     1,536        101.7  void raft::linalg::detail::reduce_cols_by_key_direct_kernel<double, thrust::transform_iterator<cuvs…
      0.0           17,056          7    2,436.6    2,560.0     2,176     2,592        169.8  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, double, double, double, double, (bool)0, …
      0.0           16,864         15    1,124.3    1,120.0       896     1,280        108.8  cupy_arange__int_int_int32                                                                          
      0.0           15,904          8    1,988.0    1,840.0     1,632     2,464        300.4  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceKern…
      0.0           15,392         11    1,399.3    1,344.0     1,312     1,536         97.2  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0           11,808          1   11,808.0   11,808.0    11,808    11,808          0.0  void cub::detail::histogram::DeviceHistogramSweepKernel<cub::detail::histogram::policy_hub<int, dou…
      0.0           11,744          8    1,468.0    1,472.0     1,344     1,600         94.4  void raft::matrix::detail::gather_kernel<raft::matrix::detail::gather_policy<(int)128, (int)1>, dou…
      0.0           11,680         10    1,168.0    1,184.0     1,120     1,184         22.6  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, long, …
      0.0           11,552          8    1,444.0    1,488.0     1,280     1,536         97.4  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.0           11,520          7    1,645.7    1,696.0     1,408     1,760        113.7  cupy_nonzero_kernel                                                                                 
      0.0           10,720          7    1,531.4    1,472.0     1,312     1,920        205.3  void cub::CUB_200800_SM_500_520_600_610_700_720_750_800_860_870_890_900_1000_1200::DeviceReduceSing…
      0.0            9,920          8    1,240.0    1,232.0     1,120     1,344         90.1  void cub::detail::for_each::static_kernel<cub::detail::for_each::policy_hub_t::policy_500_t, int, c…
      0.0            9,536          2    4,768.0    4,768.0     4,736     4,800         45.3  cupy_concatenate                                                                                    
      0.0            8,992          8    1,124.0    1,136.0       928     1,312        164.5  void cub::detail::scan::DeviceCompactInitKernel<cub::ScanTileState<int, (bool)1>, int *>(T1, int, T…
      0.0            6,848          3    2,282.7    2,208.0     2,208     2,432        129.3  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (…
      0.0            6,272          1    6,272.0    6,272.0     6,272     6,272          0.0  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, double, double, double, double, (…
      0.0            4,768          2    2,384.0    2,384.0     2,336     2,432         67.9  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0            4,544          2    2,272.0    2,272.0     2,176     2,368        135.8  void cub::detail::reduce::DeviceReduceKernel<cub::detail::reduce::policy_hub<double, unsigned int, …
      0.0            3,072          2    1,536.0    1,536.0     1,504     1,568         45.3  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0            2,848          2    1,424.0    1,424.0     1,280     1,568        203.6  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            2,592          2    1,296.0    1,296.0     1,152     1,440        203.6  cupy_copy__int_float64                                                                              
      0.0            1,856          1    1,856.0    1,856.0     1,856     1,856          0.0  void cub::detail::reduce::DeviceReduceSingleTileKernel<cub::detail::reduce::policy_hub<double, unsi…
      0.0            1,568          1    1,568.0    1,568.0     1,568     1,568          0.0  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            1,376          1    1,376.0    1,376.0     1,376     1,376          0.0  void cub::detail::transform::transform_kernel<cub::detail::transform::policy_hub<(bool)1, cuda::std…
      0.0            1,344          1    1,344.0    1,344.0     1,344     1,344          0.0  void cub::detail::histogram::DeviceHistogramInitKernel<(int)1, double, int>(cuda::std::__4::array<i…
      0.0            1,216          1    1,216.0    1,216.0     1,216     1,216          0.0  cupy_copy__int_int32                                                                                

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)   Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  -------  --------  --------  --------  --------  -----------  ------------------------------
     85.4      151,074,144  101,035   1,495.3   1,408.0     1,312     3,328        221.5  [CUDA memcpy Device-to-Host]  
     14.4       25,551,488   23,388   1,092.5   1,120.0       832     1,504        119.3  [CUDA memcpy Device-to-Device]
      0.1          202,144      174   1,161.7   1,120.0       736     4,928        445.7  [CUDA memset]                 
      0.1          159,680      147   1,086.3     928.0       704     4,192        518.8  [CUDA memcpy Host-to-Device]  

Processing [gpu_adaptive_clustering_profile.sqlite] with [/home1/apps/nvidia/Linux_aarch64/24.7/profilers/Nsight_Systems/host-linux-armv8/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  -------  --------  --------  --------  --------  -----------  ------------------------------
     34.304      174     0.197     0.002     0.000    12.976        1.396  [CUDA memset]                 
      6.933   23,388     0.000     0.000     0.000     0.080        0.001  [CUDA memcpy Device-to-Device]
      0.808  101,035     0.000     0.000     0.000     0.016        0.000  [CUDA memcpy Device-to-Host]  
      0.390      147     0.003     0.000     0.000     0.033        0.007  [CUDA memcpy Host-to-Device]  

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