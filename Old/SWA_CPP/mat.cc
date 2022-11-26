
#include <iostream>

using std::cout;
using std::endl;

template <typename T>
struct Mat{
    T** mat;
    short rows;
    short cols;
    
    short i,j;

    Mat() {}
    Mat(T** m, short r, short c): mat{m}, rows{r}, cols{c} {} 

    ~Mat() {
        if(mat != nullptr){
            for(i= 0; i<rows; i++) delete [] mat[i];
            delete [] mat;
            mat = nullptr;
        }

        }

    T** get_data(){
        T** new_mat = new T*[rows];
            for(i = 0; i< rows; i++) {
                new_mat[i] = new short[cols];
                for(j=0; j< cols; j++) new_mat[i][j] = mat[i][j];
                }
        return new_mat;
    }

    void copy_from(Mat<T>  &m){
        rows = m.rows;
        cols = m.cols;
        mat = new T*[rows];
        for(i = 0; i< rows; i++) {
            mat[i] = new short[cols];
            for (j=0; j<cols; j++) mat[i][j] = m[i][j];
        }

    }


    Mat<T>& operator= (Mat<T> &rhs) {
        for (i=0; i<rows; i++){
            for (j=0; j<cols; j++) mat[i][j] = rhs[i][j];
        }
        return *this;
    }

    T* operator[](short index){
        return mat[index];
    }





};


template <typename T>
std::ostream& operator<<( std::ostream &os, Mat<T> &m) {
        for(int i = 0; i< m.rows; i++){
            for(int j = 0; j< m.cols; j++) cout<<m.mat[i][j]<<" ";
        cout<<endl;
    }
    cout<<endl;
        return os;
    }