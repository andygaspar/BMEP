

class InputHandler{
private:
	double **dist;
	ifstream in;
	bool distort;

public:
	InputHandler(){}
	~InputHandler(){}
	// void DoubleStochasticRescaling(int n, double **, double** &);


	void StabilityChecker(int, double **);
	void MachineEpsilon();
	void EntropyAnalysis(int, double ** &, bool);
	void DistanceDistorter(int, double ** &);
	double** set_d(double*, int);
};

double** InputHandler::set_d(double* d, int n_taxa){
	dist=new double*[n_taxa+1]; for(register unsigned int i=0; i<=n_taxa;i++) dist[i] = new double[n_taxa+1];
	for (int i=0; i< n_taxa + 1; i++) dist[0][i] = 0;  
	for (int i=1; i< n_taxa + 1; i++) {
		dist[i][0] = 0;
		for (int j=1; j< n_taxa + 1; j++) dist[i][j] = d[(i-1)*n_taxa + j -1];

	}
	return dist;
}

void InputHandler::MachineEpsilon(){while ((1+EPS) != 1){TOLL = EPS; EPS /=2;} cout << "Machine epsilon: "<< TOLL << endl;} 
void InputHandler::StabilityChecker(int n, double **d){
	double m = INF;
	int acc=0;
	double smallest=INF;
	for(int i=1; i<=n-1; i++){
		for(int j=i+1; j<=n; j++){
			if(m > log2(d[i][j]/EPS) && log2(d[i][j]/EPS) >0){
				m=log2(d[i][j]/EPS);	
				// cout<<"d("<<i<<","<<j<<")="<<d[i][j]<<"\t"<<"EPS: "<<EPS<<"\tValue: "<<m<<endl;
			}
			if(d[i][j] < 1/pow(10.0,10.0)) acc++;
			if(smallest > d[i][j]) smallest=d[i][j];
		}
	}
	cout<<"The smallest entry in the matrix has value "<<smallest<<"."<<endl;
	cout<<"The matrix contains "<<acc<<" number of entries having value smaller than 10e-10."<<endl;
	cout<<"Numerical stability issues for the considered instance concerns path-lengths longer than or equal to "<<floor(m)<<"."<<endl;	
}

void InputHandler::DistanceDistorter(int n, double ** &D){
	if(n <= 6){cout<<"less than 7 taxa, performing no distorsion..."<<endl; return;}
	cout<<"Priting D before distorsion..."<<endl;
	cout<<endl;
	//finding the maximum distance in the matrix...
	double max_dist=-1; 
	for(int i=1; i<=n-1; i++) for(int j=i+1; j<=n; j++) if(D[i][j]>max_dist) max_dist=D[i][j];
	
	//Taxon per taxon, we now set the last 5 largest distances to the maximum distance
	struct PAIR{int taxon; double distance;};
	for(int i=1; i<=n; i++){
		//We use vector v as an auxiliary variable to sort the entries of row i of D
		vector<PAIR> v;
		for(int j=1; j<=n; j++) if(j!=i){ PAIR p; p.taxon=j; p.distance=D[i][j]; v.push_back(p); }

		// cout<<"printing v vector..."<<endl;	
		// for(int j=0; j<n-1; j++){
		// 	cout<<v[j].taxon<<"\t"<<v[j].distance<<endl;			
		// }	
		
		//cout<<"sorting v..."<<endl;
		//Sorting the entries of row i of D
		for(int r=0; r<n-2; r++){
			for(int s=r+1; s<n-1; s++){
				if(v[s].distance < v[r].distance){
					int t = v[r].taxon; double val = v[r].distance; 
					v[r].taxon = v[s].taxon; v[r].distance = v[s].distance;
					v[s].taxon = t; v[s].distance = val;
				}
			}
		}
		//cout<<"done!"<<endl;
		//cout<<"applying distortion to D..."<<endl;
     	//Now we set the relative entries to max distance
     	max_dist=10000;
		for(int k=v.size()-1; k>=6; k--){
			int t = v[k].taxon; 
			D[t][i]= D[i][t] = max_dist; 
		}
		//cout<<"done!"<<endl;
		v.clear();
	}	

}


void InputHandler::EntropyAnalysis(int n, double ** &D, bool rescaling){
	double **d=new double*[n+1]; for(register unsigned int i=0; i<=n; i++) d[i] = new double[n+1];
	for(int i=1; i<=n; i++) for(int j=1; j<=n; j++) d[i][j]=D[i][j];

	//PUTTING d in double stochastic form
	for(register unsigned int i=1;i<=n;++i)for(register unsigned int j=1;j<=n;++j)if(i!=j)d[i][j]=pow(2.0,-d[i][j]);
	double* rows=new double[n+1];
	double* cols=new double[n+1];
	for(register unsigned int i=1;i<=n;++i){rows[i]=0.0;cols[i]=0.0;for(register unsigned int j=1;j<=n;++j)if(i!=j){rows[i]+=d[i][j];cols[i]+=d[j][i];}}		
	double mean=0.0;
	for(register unsigned int i=1;i<=n;++i)mean+=rows[i];mean/=n;
	double TOLL1 = 0.0000000001;	
	while(true){
		int p=1,q=1;for(register unsigned int i=1;i<=n;i++){if(abs(rows[i]-mean)>abs(rows[p]-mean))p=i;if(abs(cols[i]-mean)>abs(cols[q]-mean))q=i;}
		double ident=rows[1];for(register unsigned int i=1;i<=n;i++)if(abs(ident-rows[i])>TOLL1||abs(ident-cols[i])>TOLL1){ident=-1;break;}
		if(abs(rows[p]-ident)<TOLL1&&abs(cols[q]-ident)<TOLL1)break;
		else if(abs(cols[q]-mean)>abs(rows[p]-mean)){
			double mean_c=0.0;for(register unsigned int i=1;i<=n;++i)if(i!=q)mean_c+=cols[i];mean_c/=(n-1)*cols[q];
			for(register unsigned int i=1;i<=n;++i)if(i!=q){rows[i]-=d[i][q];d[i][q]*=mean_c;rows[i]+=d[i][q];}
			cols[q]*=mean_c;
		}else{
			double mean_r=0.0;for(register unsigned int i=1;i<=n;++i)if(i!=p)mean_r+=rows[i];mean_r/=(n-1)*rows[p];
			for(register unsigned int i=1;i<=n;++i)if(i!=p){cols[i]-=d[p][i];d[p][i]*=mean_r;cols[i]+=d[p][i];}
			rows[p]*=mean_r;
		}
	}
	for(register unsigned int i=1;i<=n-1;i++){
		for(register unsigned int j=i+1;j<=n;j++){
			d[i][j]=(-log(d[i][j]/rows[1])/log(2));
			d[j][i]=d[i][j];
		}	
	}
	//now d is in double stochastic form
	double third_eq = 0; 
	for(int i=1; i<=n; i++){
		for(int j=1; j<=n; j++){
			if(i!=j){
				third_eq += d[i][j] / pow(2.0, d[i][j]); 	
			}
		}
	}

	double ratio = third_eq / (2*n-3); 
	cout<<"\nThe entropic ratio for the given distance matrix is: "<<ratio<<endl<<endl;

	if(rescaling==true){
		cout<<"Rescaling distance matrix...";
		for(int i=0; i<=n; i++) for(int j=0; j<=n; j++) D[i][j]=d[i][j];
		cout<<"done."<<endl;
	}		
	else cout<<"Assuming no rescaling of the input distance matrix. "<<endl;
	for(register unsigned int i=0; i<n+1;i++) delete [] d[i]; delete [] d;
	delete[] rows;
	delete[] cols;
	if(distort==true) DistanceDistorter(n,D);
	// exit(0);
}

/*Stat rosa pristina nomine 
  Nomina nuda tenemus */