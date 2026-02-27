#include <opencv2/opencv.hpp>
int main(){
    std::string path = "C:/Users/Administrator/Desktop/aaac.tiff";
    cv::Mat tiff = cv::imread(path,cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    std::cout<<"ch:"<<tiff.channels()<<std::endl;
}