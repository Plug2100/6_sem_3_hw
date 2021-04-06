test: main_new.cxx
	mpixlcxx -O3 -qstrict main.cxx -o main
	for n in 1 2 4 8 16 32 64 128; do \
		mpisubmit.bg -n $$n -m SMP -w 00:15:00 -e "OMP_NUM_THREADS=1" main n 25 file_write ans.bin ; \
    	done
