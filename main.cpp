#include "mpi.h"
#include "ImageReader.h"
#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>

using namespace std;

vector<int> localHistogram(768);
vector<int> notSortedRankData;
vector<int> localImageValues;
vector<int> localValues;
vector<float> localPercentages(768);
//float currentRankPercentages[768];
//float localPercentages[768];
ImageReader *ir;
int resolution;
int localDimensionsProduct;
int argc;
char **argv;

void  convertToPercentages(vector<int> values){
	for(int i = 0; i<768; i++){
		localPercentages.at(i) = (float)values.at(i)/(localDimensionsProduct/3.0);
	}
}

//Description: returns vector containing all color values [red values,green values,blue values]
void StoreDemensionsAndData(const cryph::Packed3DArray<unsigned char>* pa, int totalRanks,int currentRank){
	resolution = ((int)pa->getDim1() * (int)pa->getDim2() * (int)pa->getDim3());
	for(int channel = 0; channel < pa->getDim3(); channel++){
		for (int row=0 ; row<pa->getDim1() ; row++){
			for (int c=0 ; c<pa->getDim2() ; c++){
				notSortedRankData.push_back(pa->getDataElement(row,c,channel)); //store all values in single vector
			}
		}
	}
	if(currentRank==0)
	{
		localValues = notSortedRankData;
		localDimensionsProduct = resolution;
	}
}

void MPIRank0(int totalRanks){
		ir = ImageReader::create(argv[1]);
		StoreDemensionsAndData(ir->getInternalPacked3DArrayImage(), totalRanks,0);
		cout << "Rank 0 process is preparing to image data and resolution to other ranks..." <<endl;
		int messageTag = 1;
		for(int i=1; i<argc-1; i++){
			ir = ImageReader::create(argv[i+1]);
			if(ir == nullptr)
				cerr << "could not open image file: "<< argv[i+1] <<'\n';
			else{
			StoreDemensionsAndData(ir->getInternalPacked3DArrayImage(),totalRanks,i);
			MPI_Send(&resolution,1, MPI_INT, i, messageTag, MPI_COMM_WORLD);
			MPI_Send(&notSortedRankData.front(),resolution, MPI_INT, i, messageTag, MPI_COMM_WORLD);
			}
		}
}
void MPINotRank0(int rank){
		MPI_Status status;
		int source = 0; // rank 0 sent my message
		int tag = 1;
		MPI_Recv(&localDimensionsProduct, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
		localValues.resize(localDimensionsProduct);
		MPI_Recv(&localValues.front(), localDimensionsProduct, MPI_INT, source, tag, MPI_COMM_WORLD, &status); //recieve rgb
		cout << "Rank " << rank<< " has recieved data and resolution from rank 0" <<endl;
}

int compareImages(int totalRanks, int currentRank, vector<float> localPercentages, vector<float> GP){
	float summation=0;
	float lowestValue = 0;
	int mostSimiliarRank = 0;
	for(int i=0; i<totalRanks; i++){
		if(i!=currentRank){
			summation = 0;
			for(int j = 0; j<768; j++){
				summation = summation + abs(localPercentages.at(j)- GP.at(j+i*768));
			}
			if(i==0){
				lowestValue = summation;
				mostSimiliarRank = i;
			}
			else if(i==1 && currentRank ==0){
				lowestValue = summation;
				mostSimiliarRank = i;
			}
			else if(summation<lowestValue){
				lowestValue = summation;
				mostSimiliarRank = i;
			}
			cout << "Rank " << currentRank << ": Difference between me and node " << i << " is " << summation << endl;
		}
	}
	return mostSimiliarRank;
}

void computeHistogram(vector<int> values,int dimensionsProduct){
	int resolution = dimensionsProduct/3; //resolution = col*row*channels, resolution is divisible by 3
	for(int i = 0; i<resolution; i++){
		int rIndex =values.at(i);
		localHistogram.at(rIndex)+=1;
	}
	for(int i = resolution; i<2*resolution; i++){
		int gIndex =values.at(i);
		localHistogram.at(gIndex+256)+=1;
	}

	for(int i = 2*resolution; i<dimensionsProduct; i++){
		int bIndex =values.at(i);
		localHistogram.at(bIndex+512)+=1;
	}
}
int main(int argc, char* argv[]){
	::argc = argc;
	::argv = argv;
	if (argc < 2)
		cerr << "Usage: " << argv[0] << " imageFileName\n";
	else{
		int rank;
		int communicatorSize;
		int totalRanks = argc-1;
		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);
		if(rank ==0)
			MPIRank0(totalRanks);
		else
			MPINotRank0(rank);
		computeHistogram(localValues,localDimensionsProduct); //computes localHistogram
		convertToPercentages(localHistogram);
		vector<float> totalPercentages(768*totalRanks);
		MPI_Allgather(&localPercentages.front(),768,MPI_FLOAT,&totalPercentages.front(),768,MPI_FLOAT,MPI_COMM_WORLD);
		int offset = rank * 256 * 3;
		for(int i = 0; i< 256*3; ++i){
			assert(localPercentages.at(i) == totalPercentages.at(offset+i));//gaurentees MPI all gather sent correctly
		}
		int mostSimiliarRank = compareImages(totalRanks,rank,localPercentages,totalPercentages);
		cout <<"Rank " << rank <<": most similiar image from rank " <<mostSimiliarRank<<endl;
		cout <<"Rank " << rank <<": total pixel count is " <<localDimensionsProduct<< endl;
		MPI_Finalize();
	}
	return 0;
}
