#include "LDMM.h"

/**************************************************************************
//Image to semi-local patch transformation.
//u: m X n mat represents the image
//px, py: the size of the patch
//scale: parameter to control the weight of local coordinate
**************************************************************************/
cv::Mat LDMM::image2patch_local(cv::Mat &u, int scale){
    cv::Mat pad_u;
    cv::copyMakeBorder(u, pad_u, 0, px, 0, py, cv::BORDER_WRAP);
    cv::Mat patch=cv::Mat::zeros(px*py+2, m*n, CV_32F);
    
    for(int i=0; i<px; i++){
	for(int j=0; j<py; j++){
	    cv::Rect rc(cv::Point(i, j), cv::Point(i+m, j+n));
	    cv::Mat temp=pad_u(rc);
	    cv::Mat tempT;
	    cv::transpose(temp, tempT);
	    tempT=tempT.reshape(0, 1);
	    float *pttempT=tempT.ptr<float>(0);
	    float *ptpatch=patch.ptr<float>(i+j*px);
	    for(int k=0; k<m*n; k++){
		ptpatch[k]=pttempT[k];
	    }
	}
    }
    
    //Fill in the last two rows
    float *pt1=patch.ptr<float>(px*py);
    float *pt2=patch.ptr<float>(px*py+1);
    for(int i=0; i<m*n; i++){
	pt1[i]=float(i%m+1)*float(scale)/float(m);
	pt2[i]=float(i/n+1)*float(scale)/float(n);
    }
    
    return patch;
}

/******************************************************************************
//Compute the affinity matrix
******************************************************************************/
Eigen::SparseMatrix<float> LDMM::weight_ann_local(cv::Mat &patch){
    int m1=patch.rows, n1=patch.cols;
    cv::transpose(patch, patch);
    
    //Build the KD Tree
    cv::flann::KDTreeIndexParams indexParams(8);
    cv::flann::Index kdtree(patch, indexParams);
    cv::Mat indices, dists;
    kdtree.knnSearch(patch, indices, dists, ms, cv::flann::SearchParams(3000));
    cv::transpose(indices, indices);
    cv::transpose(dists, dists);
    
    std::vector<float> sigma(n1);
    for(int i=0; i<n1; i++){
	sigma[i]=1.0/dists.at<float>(ms_normal-1, i);
    }
    
    //Row index
    std::vector<std::vector<int> > id_row(ms, std::vector<int>(n1, 0));
    for(int i=0; i<ms; i++){
	for(int j=0; j<n1; j++){
	    id_row[i][j]=j;
	}
    }

    //Column index
    std::vector<std::vector<int> > id_col(ms, std::vector<int>(n1, 0));
    for(int i=0; i<ms; i++){
	for(int j=0; j<n1; j++){
	    id_col[i][j]=indices.at<int>(i, j);
	}
    }
    
    //Weight W1
    std::vector<std::vector<float> > W1(ms, std::vector<float>(n1, 0.0));
    for(int i=0; i<ms; i++){
	for(int j=0; j<n1; j++){
	    W1[i][j]=std::exp(-std::pow(dists.at<float>(i, j)*sigma[j], 2));
	}
    }
    
    //Assemble the weights into matrix W
    Eigen::SparseMatrix<float> W(n1, n1);
    W.reserve(ms*n1);
    std::vector<Eigen::Triplet<float> > tripletList;
    for(int i=0; i<ms; i++){
	for(int j=0; j<n1; j++){
	    tripletList.push_back(Eigen::Triplet<float>(id_row[i][j], id_col[i][j], W1[i][j]));
	}
    }
    W.setFromTriplets(tripletList.begin(), tripletList.end());
    return W;
}

/******************************************************************************
//Major swap
//Swap a vector into the following index [(n-1)*m:n*m-1, 0:(n-1)*m-1]
******************************************************************************/
std::vector<int> LDMM::major_swap(std::vector<int> &p){
    std::vector<int> q(p.size());
    std::copy(p.begin()+(n-1)*m, p.end(), q.begin());
    std::copy(p.begin(), p.begin()+(n-1)*m, q.begin()+m);
    return q;
}

/******************************************************************************
//Minor swap
//Swap a vector into the following index [m-1, 0:(m-2), 2*m-1, m:(2*m-2), ...]
******************************************************************************/
std::vector<int> LDMM::minor_swap(std::vector<int> &p){
    std::vector<int> q(p.size());
    for(int i=0; i<n; i++){
	q[i*m]=p[i*m+m-1];
	std::copy(p.begin()+i*m, p.begin()+i*m+m-1, q.begin()+i*m+1);
    }
    return q;
}

/******************************************************************************
//Assemble weights, do row and column swap
******************************************************************************/
Eigen::SparseMatrix<float> LDMM::assemble_weight(Eigen::SparseMatrix<float> &W){
    int r=m*n;
    std::vector<int> idx(r);
    std::vector<int> major_idx(r);
    for(int i=0; i<r; i++){
	major_idx[i]=i;
    }
    Eigen::SparseMatrix<float> W_new(r, r);
    W_new.reserve(40*r);
    
    int iter=0;
    for(int jj=0; jj<py; jj++){
	if(jj>0){
	    major_idx=major_swap(major_idx);
	}
	
	for(int ii=0; ii<px; ii++){
	    if(ii==0){
		idx=major_idx;
	    }else{
		idx=minor_swap(idx);
	    }
	    
	    Eigen::SparseMatrix<float> left(r, r), right(r, r);
	    left.reserve(r); right.reserve(r);
	    std::vector<Eigen::Triplet<float> > tripletList1, tripletList2;
	    for(int i=0; i<r; i++){
		tripletList1.push_back(Eigen::Triplet<float>(i, idx[i], 1.0));
		tripletList2.push_back(Eigen::Triplet<float>(idx[i], i, 1.0));
	    }
	    left.setFromTriplets(tripletList1.begin(), tripletList1.end());
	    right.setFromTriplets(tripletList2.begin(), tripletList2.end());
	    W.makeCompressed(); left.makeCompressed(); right.makeCompressed();
	    W=W.pruned(1.0, 1.0e-3);
	    clock_t begin=clock();
	    Eigen::SparseMatrix<float> W_temp=left*W*right;
	    clock_t end=clock();
	    double elapsed_secs=double(end-begin)/CLOCKS_PER_SEC;
	    std::cout<<"Time taken: "<<elapsed_secs<<" Iteration: "<<iter++<<std::endl;
	    W_new+=W_temp;
	}
    }
    return W_new;
}

/******************************************************************************
//Calculate PSNR
******************************************************************************/
double LDMM::getPSNR(cv::Mat &src, cv::Mat &dest){
    double sse, mse, psnr;
    sse=0.0;
    for(int i=0; i<src.rows; i++){
	float *d=dest.ptr<float>(i);
	float *s=src.ptr<float>(i);
	for(int j=0; j<src.cols; j++){
	    sse+=(d[j]-s[j])*(d[j]-s[j]);
	}
    }
    if(sse==0.0){
	return 0.0;
    }else{
	mse=sse/(double)(src.cols*src.rows);
	//psnr=10.0*log10((255*255)/mse);
	psnr=10.0*log(1/mse)/log(10.0);
	return psnr;
    }
}


/******************************************************************************
//Low dimensional manifold model
//y: sampled image, to be inpainted
//mask: the indicator function. 0: unsampled pixel; 1: sampled pixel
//H: ground truth image
//mu: weights in the graph Laplacian
//outerloop: number of iterations to be performed for image inpainting
//px, py: patch size
//ms, ms_normal: number of neighors used to construct graph Laplacian, and for normalization
//scale_initial, scale_target: weights of the local coordinate in the semi-local patch
//out: the result images
//psnr: psnr between inpainted and ground truth
******************************************************************************/
std::pair<std::vector<cv::Mat>, std::vector<double> > LDMM::LDMM2D(){
    std::vector<cv::Mat> out;
    std::vector<double> psnr;
    //Set random seed
    cv::RNG rng;
    cv::theRNG().state=1;

    int m=y.rows, n=y.cols;
    int r=m*n;

    //Note for reshape function, openCV is row by row; Matlab is column by column.
    //Hence we first need transpose, when finished transpose again
    cv::transpose(mask, mask);
    mask=mask.reshape(0, 1);
    cv::transpose(y, y);
    y=y.reshape(0, 1);
    
    float *ptmask=mask.ptr<float>(0);
    float *pty=y.ptr<float>(0);
    
    int numNonZero=cv::countNonZero(mask);
    int numZero=r-numNonZero;
    
    //u: a copy of the sampled image in Eigen vector format
    Eigen::VectorXf u(r);
    for(int i=0; i<r; i++)
	u[i]=pty[i];
    
    //mask, ~mask
    Eigen::VectorXi idNonZero(numNonZero), idZero(numZero);
    Eigen::VectorXi rowid(1);
    rowid(0)=0;
    int countZero=0, countNonZero=0;
    for(int i=0; i<r; i++){
	if(ptmask[i]==0)
	    idZero[countZero++]=i;
	else
	    idNonZero[countNonZero++]=i;
    }
    Eigen::VectorXf y_small;
    igl::slice(u, idNonZero, rowid, y_small);
    
    float meanPixel=y_small.mean();
    float stdPixel=0.0;
    for(int i=0; i<y_small.size(); i++){
	stdPixel+=(y_small[i]-meanPixel)*(y_small[i]-meanPixel);
    }
    stdPixel/=(y_small.size());
    stdPixel=std::sqrt(stdPixel);
    std::cout<<"Mean pixel: "<<meanPixel<<std::endl;
    std::cout<<"Std pixel: "<<stdPixel<<" "<<y_small.size()<<std::endl;
    for(int i=0; i<numZero; i++){
	u[idZero[i]]=meanPixel+2.0*stdPixel*rng.gaussian(1.0);
    }
    
    Eigen::VectorXf x;
    igl::slice(u, idZero, rowid, x);

    //Starts the main loop of LDMM
    int scale=scale_initial+1;
    for(int iter=0; iter<outerloop; iter++){
	scale=std::max(scale-1, scale_target);
	//Generate the patch set
	cv::Mat uu=cv::Mat::zeros(1, r, CV_32F);
	float *ptuu=uu.ptr<float>(0);
	for(int i=0; i<r; i++){
	    ptuu[i]=u[i];
	}
	uu=uu.reshape(0, n);
	cv::transpose(uu, uu);
	
	cv::Mat patch=image2patch_local(uu, scale);
	
	//Compute the weight matrix
	Eigen::SparseMatrix<float> W=weight_ann_local(patch);
	//Get max(W, W^T)
	Eigen::SparseMatrix<float> WT(W.transpose());
	Eigen::SparseMatrix<float> A=(W+WT)/2.0;
	Eigen::SparseMatrix<float> B=(WT-W)/2.0;
	for(int k=0; k<B.outerSize(); ++k){
	    for(Eigen::SparseMatrix<float>::InnerIterator it(B, k); it; ++it){
		if(it.value()<0)
		    B.coeffRef(it.row(), it.col())=-it.value();
	    }
	}
	W=A+B;
	W=W.pruned(1.0, 0.001);
	W.makeCompressed();
	
	//Assemble weights
	W=assemble_weight(W);
	
	//Generate Graph Laplacian, L
	//Vector of row sum of the affinity matrix W, each element is the sum of the corresponding row
	Eigen::VectorXf DV=W*Eigen::VectorXf::Ones(W.cols());
	Eigen::SparseMatrix<float> D=Eigen::SparseMatrix<float>(r, r);
	D.reserve(r);
	std::vector<Eigen::Triplet<float> > tripletListD;
	for(int i=0; i<r; i++){
	    tripletListD.push_back(Eigen::Triplet<float>(i, i, DV[i]));
	}
	D.setFromTriplets(tripletListD.begin(), tripletListD.end());
	
	//Graph Laplacian
	Eigen::SparseMatrix<float> L=D-W;
	
	//Set up linear equations
	//WM=W(~mask, mask), LM=L(~mask, mask). These are used to contruct Mat_2
	Eigen::SparseMatrix<float> WM(numZero, numNonZero);
	Eigen::SparseMatrix<float> LM(numZero, numNonZero);
	igl::slice(W, idZero, idNonZero, WM);
	igl::slice(L, idZero, idNonZero, LM);
	Eigen::SparseMatrix<float> Mat_2=mu*WM-2.0*LM;
	
	//Mat_1, LN=L(~mask, ~mask)
	Eigen::VectorXf deltaV=WM*Eigen::VectorXf::Ones(WM.cols());
	Eigen::SparseMatrix<float> delta=Eigen::SparseMatrix<float>(numZero, numZero);
	std::vector<Eigen::Triplet<float> > tripletListdelta;
	for(int i=0; i<numZero; i++){
	    tripletListdelta.push_back(Eigen::Triplet<float>(i, i, deltaV[i]));
	}
	delta.setFromTriplets(tripletListdelta.begin(), tripletListdelta.end());
	Eigen::SparseMatrix<float> LN(numZero, numZero);
	igl::slice(L, idZero, idZero, LN);
	Eigen::SparseMatrix<float> Mat_1=2.0*LN+mu*delta;
	
	//The right hand side of the linear system
	Eigen::VectorXf b=Mat_2*y_small;
	//Unsampled region: to solve through the linear system
	Eigen::VectorXf x(numZero);
	//Solve the linear system
	Eigen::BiCGSTAB<Eigen::SparseMatrix<float> > Solver;
	Solver.compute(Mat_1);
	x=Solver.solve(b);
	//Fill in the missing part of u
	int countx=0;
	for(int i=0; i<numZero; i++){
	    u[idZero[i]]=x[countx++];
	}
	
	for(int i=0; i<r; i++){
	    if(u[i]<0 || u[i]>1)
		std::cout<<"Bad value: "<<u[i]<<std::endl;
	}
	
	//Convert u to an image
	cv::Mat img=cv::Mat::zeros(1, r, CV_32F);
	float *ptimg=img.ptr<float>(0);
	for(int i=0; i<r; i++)
	    ptimg[i]=u[i];
	img=img.reshape(0, n);
	cv::transpose(img, img);
	out.push_back(img);
	
	cv::imshow("img", img);
	
	//Compute PSNR
	psnr.push_back(getPSNR(img, H));
	std::cout<<"PSNR: "<<psnr.back()<<std::endl;
    }
    cv::waitKey();
    return std::make_pair(out, psnr);
}
