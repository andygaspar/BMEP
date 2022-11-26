
#include <iostream>
#include "mat.cc"
using std::cout;
using std::endl;


template <typename T>
void print_mat(T** d, int rows, int cols=0){
    if(cols == 0) cols = rows;
    for(int i = 0; i< rows; i++){
        for(int j = 0; j< cols; j++) cout<<d[i][j]<<" ";
        cout<<endl;
    }
    cout<<endl;
}


double init_d[] = {0.0,0.0204987,0.0184985,0.0246588,0.0266809,0.0367091,
                0.0204987,0.0,0.0142995,0.0176905,0.0195088,0.0299465,
                0.0184985,0.0142995,0.0,0.0190038, 0.0191878, 0.0276666,
                0.0246588, 0.0176905, 0.0190038, 0.0, 0.0258093, 0.0368426,
                0.0266809, 0.0195088, 0.0191878, 0.0258093, 0.0, 0.0354803,
                0.0367091, 0.0299465, 0.0276666, 0.0368426, 0.0354803, 0.0};


short n_taxa = 6;
short m = 10;



double** get_d(){
    double** d = new double*[n_taxa];
    for(int i = 0; i < n_taxa; i++) {
    d[i] = new double[n_taxa];
    for(int j = 0; j<n_taxa; j++) d[i][j] = init_d[i*n_taxa + j];
    }
    return d;
}




