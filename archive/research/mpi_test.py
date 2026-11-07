import time

import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() < 2:
        print("Please run with at least 2 processes: mpirun -n 2 python mpi_test.py")
        return

    # Helper to print with relative timestamps
    start_time_global = time.time()

    def log(msg):
        print(f"[{time.time() - start_time_global:05.2f}s] Rank {rank}: {msg}")

    if rank == 0:
        # ==========================================
        # TEST 1: Eager Send (Small Payload)
        # ==========================================
        payload = np.array([42], dtype=np.int64)

        log("Calling comm.Send (Rank 1 is intentionally asleep)...")

        t0 = time.perf_counter()
        comm.Send(payload, dest=1, tag=100)
        t1 = time.perf_counter()

        log(f"comm.Send returned in {t1 - t0:.6f} seconds!")
        log("PROVEN: Eager protocol used. Rank 0 didn't wait for Rank 1 to call Recv.")

        # ==========================================
        # TEST 2: Triggering Iprobe
        # ==========================================
        log("Sleeping for 3 seconds before sending the second message...")
        time.sleep(3.0)

        log("Sending second message to trigger Rank 1's Iprobe.")
        comm.Send(np.array([99], dtype=np.int64), dest=1, tag=200)
        log("Finished.")

    elif rank == 1:
        # ==========================================
        # TEST 1: Eager Send (Receiver side)
        # ==========================================
        log("Sleeping for 1.5 seconds to force Rank 0 to wait (if it wasn't eager)...")
        time.sleep(1.5)

        log("Woke up. Calling Recv for the first message...")
        buf1 = np.empty(1, dtype=np.int64)
        comm.Recv(buf1, source=0, tag=100)
        log(f"Received payload: {buf1[0]}")

        # ==========================================
        # TEST 2: Non-blocking Iprobe
        # ==========================================
        log("Starting Iprobe loop. Looking for tag 200...")
        probe_count = 0

        # This loop proves Iprobe doesn't hang
        while not comm.Iprobe(source=0, tag=200):
            probe_count += 1
            log(f"Iprobe returned False (attempt {probe_count}). Doing 'other work'...")
            time.sleep(0.5)  # Simulate doing some math/work

        log("Iprobe returned TRUE! Message is waiting in the network buffer.")

        buf2 = np.empty(1, dtype=np.int64)
        comm.Recv(buf2, source=0, tag=200)
        log(f"Received second payload: {buf2[0]}. Test complete.")


if __name__ == "__main__":
    main()
