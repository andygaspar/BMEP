#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <omp.h>
// #include "test.h"

using std::cout;
using std::endl;

class Swa{
    public:

    double** d;
    short** adj_mat;
    short** current_mat;
    short** Tau;
    short** tau_temp;
    short** min_adj_mat;

    double* powers;

    short n_taxa;
    short m, n_internals;

    double min_val;
    double best_val, current_val;

    
    short tax_edge, int_edge, size_reduced;
    short int_node, new_node, i, j;

    


    Swa(double** d_input, short n_taxa_input):
        d{d_input}, n_taxa{n_taxa_input}{

            m = n_taxa * 2 - 2;
            n_internals = m - n_taxa;

            min_val = 1000;
            best_val = 1000;

            adj_mat = init_adj();

            current_mat = new short*[m]; 
            min_adj_mat = new short*[m];

            Tau = new short*[m];
            for(i = 0; i< m; i++) Tau[i] = new short[m];

            tau_temp = new short*[m];
            for(i = 0; i< m; i++) tau_temp[i] = new short[m];

            for(i = 0; i<m; i ++) {
                current_mat[i] = new short[n_internals];
                min_adj_mat[i] = new short[n_internals];
            }
            powers = new double[n_taxa];
            for(i=0; i<=n_taxa; i++) powers[i]=1/std::pow(2.0,i);

    }

    ~Swa () {
        for(i=0; i<m; i++) {delete [] adj_mat[i]; delete [] min_adj_mat[i]; delete [] current_mat[i]; delete [] Tau[i]; delete [] tau_temp[i];}
        delete [] adj_mat; delete [] min_adj_mat; delete [] current_mat; delete [] Tau; delete [] tau_temp;

    }

    short** init_adj(){
        short** adj_mat = new short*[m];
        for(int i = 0; i < m; i++) {
            adj_mat[i] = new short[n_internals];
            for(j=0; j<n_internals; j++) adj_mat[i][j] = 0;
        }
        adj_mat[0][0] = 1; adj_mat[1][0] = 1; adj_mat[2][0] = 1;
        return adj_mat;

    }


    short** add_node(short** curr_mat, short idx_0,  short idx_1, short step, short new_n){
        // print_mat(curr_mat, m, n_internals);
        curr_mat[idx_0][ idx_1] = 0;  // detach selected
        curr_mat[idx_0][new_n - n_taxa] = 1; //reattach selected to new
        curr_mat[step][new_n - n_taxa] = 1;
        curr_mat[idx_1 + n_taxa][new_n - n_taxa] = 1;   // reattach selected to new
        // print_mat(curr_mat, m, n_internals);
        return curr_mat;
    }


    double compute_val(short** Tau, short** curr_mat, short step, short size, short num_current_internals){
        step ++;
        for(i=0; i<size; i++) {
            for(j=0; j<size; j++) Tau[i][j] = size;
            Tau[i][i]=0;
        }

        // print_mat(current_mat, 9, 4);
                short iter, k;

        for(i=0; i < step; i++) {
            for(j=0; j<n_internals; j++) {
                if(curr_mat[i][j] != 0) {
                    Tau[i][j + step] = 1;
                    Tau[j + step][i] = 1;
                }
            }
        }
        // print_mat(Tau, size);
        for(i=0; i < num_current_internals; i++) {
            for(j=0; j<n_internals; j++) {
                if(curr_mat[i + n_taxa][j] != 0) {
                    Tau[i + step][j + step] = 1;
                    Tau[j + step][i + step] = 1;
                }
            }
        }
        // print_mat(Tau, size);


        for(iter=0; iter< size; iter++){
            for(i=0; i<size; i++){
                for(j=0; j<size; j++) tau_temp[i][j] = Tau[iter][i] + Tau[j][iter];

            }
            for(i=0; i<size; i++){
                for(j=0; j<size; j++) {
                    if(tau_temp[i][j]< Tau[i][j]) Tau[i][j] = tau_temp[i][j];
                }
            }
        }

        current_val = 0;
        for(i=0; i<step; i++){
                for(j=i+1; j<step; j++) current_val += d[i][j]*powers[Tau[i][j]];
        }
        
        // print_mat(Tau, size);
        

        return current_val*2;
    }
    
    void solve(){
        short n_edges = 3;
        short num_internals;
        short selection;
        for(short step=3; step < n_taxa; step++){
            best_val = 100;
            // cout<<"step ****"<<step<<endl;
            new_node = n_taxa + step - 2;
            size_reduced = (step + 1) * 2 - 2;
            
            num_internals = step - 1; // is added 1 as in compute val step is increased, however is common to all loops

            for(tax_edge=0; tax_edge<step;tax_edge++){

                selection = 0;
                while(adj_mat[tax_edge][selection]!=1) selection++;

                for(i=0; i<m; i++) {for(j=0; j<n_internals; j++) current_mat[i][j] = adj_mat[i][j];}

                
                current_mat = add_node(current_mat, tax_edge,  selection, step, new_node);
                current_val = compute_val(Tau, current_mat, step, size_reduced, num_internals);
                if(current_val< best_val){
                    best_val = current_val;
                    for(i=0; i<m; i++) {for(j=0; j<n_internals; j++) min_adj_mat[i][j] = current_mat[i][j];}
                }

            }
            
            for(int_edge=n_taxa; int_edge < num_internals ;int_edge++){
                for(selection = int_edge - n_taxa + 1; selection < n_internals; selection++){
                    if(adj_mat[int_edge][i] != 0){

                        for(i=0; i<m; i++) {for(j=0; j<n_internals; j++) current_mat[i][j] = adj_mat[i][j];}
                        current_mat = add_node(current_mat, tax_edge,  selection, step, new_node);
                        current_val = compute_val(Tau, current_mat, step, size_reduced, num_internals);
                        if(current_val< best_val){
                            best_val = current_val;
                            for(i=0; i<m; i++) {for(j=0; j<n_internals; j++) min_adj_mat[i][j] = current_mat[i][j];}
                            }
                    }

                }
                
            }
            // print_mat(min_adj_mat, m, n_internals);
            for(i=0; i<m; i++) {for(j=0; j<n_internals; j++) adj_mat[i][j] = min_adj_mat[i][j];}

        }
    }


};





// int main() {
//     double** d = get_d();
//     Swa swa{d, 6};

//     swa.solve();
//     cout<<"here  best "<<swa.best_val<<endl;
// }

extern "C" {

    double swa_(double* dd, short n_taxa) {
            double ** d = new double*[n_taxa];
            for(int i=0; i< n_taxa ; i++) d[i] = &dd[i*n_taxa];
            Swa swa{d, n_taxa};

            swa.solve();
            cout<<"here  best "<<swa.best_val<<endl;
            d=nullptr;
            return swa.best_val;
        }

    void print_(double* dd, short size){
    for(int i = 0; i< size; i++){
        for(int j = 0; j< size; j++) cout<<dd[i*size + j]<<" ";
        cout<<endl;
    }
    cout<<endl;
    } 



}