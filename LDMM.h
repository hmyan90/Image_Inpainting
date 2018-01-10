//OpenCV Utilities
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//Eigen Utilities
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
//STL Utilities
#include <iostream>
#include <vector>
#include <fstream>
//igl Utilities
#include <igl/slice.h>

class LDMM{
private:
    cv::Mat H;		//Ground truth image
    cv::Mat y;		//The sampled image
    cv::Mat mask;	//Indicator of the sampled region
    int m, n;		//The size of the image
    int px, py;		//Patch size
    float rate;		//Sampling rate
    float mu;		//Weight in the weight graph Laplacian
    //ms: the number of neighbors involved in constructing graph Laplacian
    //ms_normal: the index of the distance used in Gaussian kernel
    int ms, ms_normal;
    //Number of iterations required
    int outerloop;
    //Weights of the local coordinate in semi-local patch
    int scale_target, scale_initial;
    
public:
    /**************************************************************************
    //Constructor
    **************************************************************************/
    LDMM(cv::Mat &H0, cv::Mat &y0, cv::Mat &mask0, int m0, int n0, int px0=10, int py0=10, float rate0=0.1, int ms0=20, int ms_normal0=10, int outerloop0=15, int scale_target0=3, int scale_initial0=10){
	H=H0; y=y0; mask=mask0;
	m=m0; n=n0;
	px=px0; py=py0;
	rate=rate0;
	mu=1.0/rate-1.0;
	ms=ms0; ms_normal=ms_normal0;
	outerloop=outerloop0;
	scale_target=scale_target0;
	scale_initial=scale_initial0;	
    }

    /**************************************************************************
    //Destructor
    **************************************************************************/
    ~LDMM(){}

    /**************************************************************************
    //Image to semi-local patch transformation.
    //u: m X n mat represents the image
    //px, py: the size of the patch
    //scale: parameter to control the weight of local coordinate
    **************************************************************************/
    cv::Mat image2patch_local(cv::Mat &u, int scale);
    
    /******************************************************************************
    //Compute the affinity matrix
    ******************************************************************************/
    Eigen::SparseMatrix<float> weight_ann_local(cv::Mat &patch);
    
    /******************************************************************************
    //Major swap
    //Swap a vector into the following index [(n-1)*m:n*m-1, 0:(n-1)*m-1]
    ******************************************************************************/
    std::vector<int> major_swap(std::vector<int> &p);
    
    /******************************************************************************
    //Minor swap
    //Swap a vector into the following index [m-1, 0:(m-2), 2*m-1, m:(2*m-2), ...]
    ******************************************************************************/
    std::vector<int> minor_swap(std::vector<int> &p);

    /******************************************************************************
    //Assemble weights, do row and column swap
    ******************************************************************************/
    Eigen::SparseMatrix<float> assemble_weight(Eigen::SparseMatrix<float> &W);

    /******************************************************************************
    //Calculate PSNR
    ******************************************************************************/
    double getPSNR(cv::Mat &src, cv::Mat &dest);
    
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
    std::pair<std::vector<cv::Mat>, std::vector<double> > LDMM2D();
};
