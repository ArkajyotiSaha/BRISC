#include <string>

void zeros(double *a, int n);

void mvrnorm(double *des, double *mu, double *cholCov, int dim);

double logit(double theta, double a, double b);

double logitInv(double z, double a, double b);

double dist2(double &a1, double &a2, double &b1, double &b2);

//Description: given a location's index i and number of neighbors m this function provides the index to i and number of neighbors in nnIndx
void getNNIndx(int i, int m, int &iNNIndx, int &iNN);

//Description: creates the nearest neighbor index given pre-ordered location coordinates.
//Input:
//n = number of locations
//m = number of nearest neighbors
//coords = ordered coordinates for the n locations
//Output:
//nnIndx = set of nearest neighbors for all n locations (on return)
//nnDist = euclidean distance corresponding to nnIndx (on return)
//nnIndxLU = nx2 look-up matrix with row values correspond to each location's index in nnIndx and number of neighbors (columns 1 and 2, respectively)
//Note: nnIndx and nnDist must be of length (1+m)/2*m+(n-m-1)*m on input. nnIndxLU must also be allocated on input.
void mkNNIndx(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);

//void mkNNIndx2(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);


//Description: using the fast mean-distance-ordered nn search by Ra and Kim 1993
//Input:
//ui = is the index for which we need the m nearest neighbors
//m = number of nearest neighbors
//n = number of observations, i.e., length of u
//sIndx = the NNGP ordering index of length n that is pre-sorted by u
//u = x+y vector of coordinates assumed sorted on input
//rSIndx = vector or pointer to a vector to store the resulting nn sIndx (this is at most length m for ui >= m)
//rNNDist = vector or point to a vector to store the resulting nn Euclidean distance (this is at most length m for ui >= m)  

double dmi(double *x, double *c, int inc);

double dei(double *x, double *c, int inc);

void fastNN(int m, int n, double *coords, int ui, double *u, int *sIndx, int *rSIndx, double *rSNNDist);

//Description: given the nnIndex this function fills uIndx for identifying those locations that have the i-th location as a neighbor.
//Input:
//n = number of locations
//m = number of nearest neighbors
//nnIndx = set of nearest neighbors for all n locations
//Output:
//uIndx = holds the indexes for locations that have each location as a neighbor
//uIndxLU = nx2 look-up matrix with row values correspond to each location's index in uIndx and number of neighbors (columns 1 and 2, respectively)
//Note: uIndx must be of length (1+m)/2*m+(n-m-1)*m on input. uINdxLU must also be allocated on input.
void mkUIndx(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU);

std::string getCorName(int i);

double spCor(double &D, double &phi, double &nu, int &covModel, double *bk);

int which(int a, int *b, int n);

double Q(double *B, double *F, double *u, double *v, int n, int *nnIndx, int *nnIndxLU);

//trees
struct Node{
	int index; // which point I am
	Node *left;
	Node *right; 
	Node (int i) { index = i; left = right = NULL; }
};

Node *miniInsert(Node *Tree, double *coords, int index, int d,int n);

void get_nn(Node *Tree, int index, int d, double *coords, int n, double *nnDist, int *nnIndx, int iNNIndx, int iNN, int check);

void mkNNIndxTree0(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);
