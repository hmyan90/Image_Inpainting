//This code implements the LDMM WGL for image inpainting
//Compile: g++ -I../LDMM_V2/libigl/include/ -I/usr/include/eigen3 *.cpp -o LDMM `pkg-config --cflags --libs opencv` -O3
#include "LDMM.h"

int main(int argc, char **argv){
    //H is the ground truth image, 0 means read grayscale image
    cv::Mat H0=cv::imread("H.png", 0);
    if(!H0.data) return -1;
    cv::Mat H;
    H0.convertTo(H, CV_32F);
    std::cout<<"H's channels: "<<H.channels()<<" H's size: "<<H.rows<<" "<<H.cols<<std::endl;
    H/=255.0;
    //y is the sampled image
    cv::Mat y0=cv::imread("y.png", 0);
    if(!y0.data) return -1;
    cv::Mat y;
    y0.convertTo(y, CV_32F);
    std::cout<<"y's channels: "<<y.channels()<<" y's size: "<<y.rows<<" "<<y.cols<<std::endl;
    y/=255.0;

    //mask is the indicator function
    cv::Mat mask0=cv::imread("mask.png", 0);
    if(!mask0.data) return -1;
    cv::Mat mask;
    mask0.convertTo(mask, CV_32F);
    std::cout<<"mask's channels: "<<mask.channels()<<" mask's size: "<<mask.rows<<" "<<mask.cols<<std::endl;
    mask/=255;
    
    //m and n are the number of pixels along each side of the image
    int m=H.rows, n=H.cols;
    //Sampling rate
    float rate=0.1;
    //Weight in the weighted graph Laplacian
    float mu=1.0/rate-1.0;
    //Patch size
    int px=10, py=10;
    //ms: the number of neighbors involved in constructing graph Laplacian
    //ms_normal: the index of the distance used in Gaussian kernel
    int ms=20, ms_normal=10;
    //Number of iterations required
    int outerloop=15;
    //Weights of the local coordinate in semi-local patch
    int scale_target=3, scale_initial=10;
    
    //Declare an LDMM object
    LDMM ldmm(H, y, mask, m, n, px, py, rate, ms, ms_normal, outerloop, scale_target, scale_initial);
    
    //Perform 2D image inpainting
    clock_t begin=clock();
    std::pair<std::vector<cv::Mat>, std::vector<double> > res=ldmm.LDMM2D();
    clock_t end=clock();
    std::cout<<"Time taken: "<<double(end-begin)/CLOCKS_PER_SEC*1000.<<" ms."<<std::endl;
    return 0;
}
