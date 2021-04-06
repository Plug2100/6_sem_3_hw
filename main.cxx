#include <mpi.h>
#include <iostream>
#include <complex>
#include <assert.h>
#include "omp.h"
#include <fstream>
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>
#define eps 0.01
using namespace std;
typedef complex<double> complexd;



double normal_dis_gen() //generate a value
{
    double S = 0.;
    for (int i = 0; i < 12; ++i) {S += (double) rand() / RAND_MAX; }
    return S - 6.0;
}



complexd *read(char *f, int rank, unsigned long long seg_size) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL,&file);

    //auto *A = new complexd[seg_size];
    complexd *A;
    A = (complexd*) malloc(sizeof(complexd) * seg_size);



    double d[2];
    MPI_File_seek(file, 2 * seg_size * rank * sizeof(double), MPI_SEEK_SET);
    for (int i = 0; i < seg_size; ++i) {
        MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
       // A[i].real() = d[0];
        //A[i].imag() = d[1];
        A[i] = complexd(d[0], d[1]);
    }
    MPI_File_close(&file);
    return A;
}








void write(char *f, complexd *B, int n, int rank, int size, unsigned long long seg_size) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    double d[2];
    MPI_File_seek(file, 2 * seg_size * rank * sizeof(double), MPI_SEEK_SET);
    for (int i = 0; i < seg_size; ++i) {
        d[0] = B[i].real();
        d[1] = B[i].imag();
        MPI_File_write(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
}










complexd* generate_condition(unsigned long long seg_size, int rank, int size){
    double module = 0;
    unsigned int seed = time(NULL) + rank;
    complexd *V;
    V = (complexd*) malloc(sizeof(complexd) * seg_size);
    for (long long unsigned  i = 0; i < seg_size; i++){
        V[i] = complexd(rand_r(&seed)%100 + 1.0, rand_r(&seed)%100 + 1.0);
      //  V[i].real() = rand_r(&seed)%100 + 1;
        //V[i].imag() = rand_r(&seed)%100 + 1;
        module += abs(V[i] * V[i]);
    }
    int rc;
    double new_m;
    MPI_Status stat;
    if(rank != 0){
        module += 1;
        rc = MPI_Send(&module, 1, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
        MPI_Recv(&module, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &stat);
    }
    else{
        for(int i = 1; i < size; i++){
            MPI_Recv(&new_m, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, &stat);
            module += new_m;
        }
        module = sqrt(module);
        for(int i = 1; i < size; i++){
            rc = MPI_Send(&module, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
        }
    }
    for (long long unsigned j = 0; j < seg_size; j++) {
        V[j] /= module;
    }
    return V;
}







void OneQubitEvolution(complexd *in, complexd *out, complexd U[2][2], int n, int q, int rank, unsigned long long seg_size) {
    int first_index = rank * seg_size;
    int rank_change = first_index ^(1u << (q - 1));     
    rank_change /= seg_size;
    if (rank != rank_change) {
        //MPI_Sendrecv(in, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, out, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int rc;
        MPI_Status stat3;
        if (rank < rank_change) {
            rc = MPI_Send(in, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD);
            MPI_Recv(out, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD, &stat3);
            for (int i = 0; i < seg_size; i++) {
                out[i] = U[0][0] * in[i] + U[0][1] * out[i];
            }
        } else {
            MPI_Recv(out, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD, &stat3);
            rc = MPI_Send(in, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD);
            for (int i = 0; i < seg_size; i++) {
                out[i] = U[1][0] * out[i] + U[1][1] * in[i];
            }
        }
    } else {
        int cr = 0;
        while(seg_size != 1){
            seg_size /= 2;
            cr++;
        }
        int shift = cr - q;
        int pow = 1 << (shift);
        for (int i = 0; i < seg_size; i++) {
            int i0 = i & ~pow;
            int i1 = i | pow;
            int iq = (i & pow) >> shift;
            out[i] = U[iq][0] * in[i0] + U[iq][1] * in[i1];
        }
    }
}











double dist(complexd *ideal, complexd *noise, int rank, unsigned long long seg_size, int size) {
    double sqr = 0;
    double hlp = 0;
    for (int i = 0; i < seg_size; i++) {
        sqr += abs(ideal[i] * conj(noise[i])) * abs(ideal[i] * conj(noise[i]));
    }
    int rc;
    MPI_Status stat2;
    if(rank != 0){
        rc = MPI_Send(&sqr, 1, MPI_DOUBLE_COMPLEX, 0, 8, MPI_COMM_WORLD);
    }
    else{
        for (int s = 1; s < size; s++){
            MPI_Recv(&hlp, 1, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, 8, MPI_COMM_WORLD, &stat2);
            sqr += hlp;   
        }
    }
    return sqr;
}









int main(int argc, char **argv) {
    int was_read = 0;
    int test = 0;
    char *input, *output, *test_file;
    unsigned k, n;
    complexd *V;
    complexd *need;
    complexd *need_new;
    for (int i = 1; i < argc; i++) { 
        string option(argv[i]);
        if (option.compare("n") == 0) {
            n = atoi(argv[++i]);
        }
        if ((option.compare("file_read") == 0)) {
            input = argv[++i];
            was_read = 1;
        }
        if ((option.compare("file_write") == 0)) {
            output = argv[++i];
        }
        if ((option.compare("test") == 0)) {
            test = 1;
        }
        if ((option.compare("file_test") == 0)) {
            test_file = argv[++i];
        }
    }
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    unsigned long long index = 1LLU << n;
    unsigned long long seg_size = index / size;
    need = (complexd*) malloc(sizeof(complexd) * seg_size);
    need_new = (complexd*) malloc(sizeof(complexd) * seg_size);
    if (was_read == 0) {
        V = generate_condition(seg_size, rank, size); 
    } else {
        V = read(input, rank, seg_size);
    }
    struct timeval start, stop;
    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);
    MPI_Status stat1;
    int error = 0;
    int rc1;
    double begin = MPI_Wtime();
    for(k = 1; k < n+1; k++){
        OneQubitEvolution(V, need, U, n, k, rank, seg_size);
    }
    complexd U_noised[2][2];
    double thetta = 0;
    for (int k = 1; k < n + 1; k++) {
        if (rank == 0) {
            thetta = normal_dis_gen();
        }
        MPI_Bcast(&thetta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        U_noised[0][0] = U[0][0] * cos(thetta) - U[0][1] * sin(thetta);
        U_noised[0][1] = U[0][0] * sin(thetta) + U[0][1] * cos(thetta);
        U_noised[1][0] = U[1][0] * cos(thetta) - U[1][1] * sin(thetta);
        U_noised[1][1] = U[1][0] * sin(thetta) + U[1][1] * cos(thetta);
        OneQubitEvolution(V, need_new, U_noised, n, k, rank, seg_size);
    }
    double end = MPI_Wtime();
    if (test == 1) {
        int rc;
        complexd *test_vector = read(test_file, rank, seg_size);
        for (int i = 0; i < seg_size; i++) {
            cout << abs(need[i] * need[i]) << endl;
            if (abs(test_vector[i].real() - need[i].real()) > 0.000001 || abs(test_vector[i].imag() - need[i].imag()) > 0.000001) {
               // cout << test_vector[i] << "   " << need[i] << endl;
                error = 1;
            }
        }
        if(rank > 0){
            rc = MPI_Send(&error, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
        else{
            MPI_Status stat;
            for(int i= 1; i < size; i++){
                MPI_Recv(&rc, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &stat);
                error += rc;
            }
            if(error != 0){
                cout << "error had been done" << endl;
            }
            else{
                cout << "no errors" << endl;
            }
        }
    } else {
        double distance = dist(need, need_new, rank, seg_size, size);
        if (rank == 0){
            cout << distance << endl;
        }
    }
    if(rank == 0){
        ofstream a;
        a.open("time_cont.txt", std::ios::app);
        a << end - begin << " " << size << endl;
        a.close();
    }
    MPI_Finalize();
    delete[] V;
    delete[] need;
    delete[] need_new;
}