// image_stitcher.cpp
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <cv_bridge/cv_bridge.h>

class ImageStitcher : public rclcpp::Node
{
public:
    ImageStitcher() : Node("image_stitcher")
    {
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ImageStitcher::stitch_images, this));
    }

private:
    void stitch_images()
    {
        cv::Mat left_img = cv::imread("/app/left.png");
        cv::Mat right_img = cv::imread("/app/right.png");

        if (left_img.empty() || right_img.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to read input images");
            return;
        }

        cv::Mat result;
        if (homography_.empty())
        {
            calibrate(left_img, right_img);
        }

        if (!homography_.empty())
        {
            cv::warpPerspective(left_img, result, homography_, cv::Size(left_img.cols + right_img.cols, right_img.rows));
            right_img.copyTo(result(cv::Rect(0, 0, right_img.cols, right_img.rows)));

            cv::imwrite("/app/stitched_output.png", result);
            RCLCPP_INFO(this->get_logger(), "Stitched image saved as 'stitched_output.png'");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute homography");
        }
    }

    void calibrate(const cv::Mat& img_1, const cv::Mat& img_2)
    {
        auto detector = cv::xfeatures2d::SIFT::create();
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;

        detector->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptors_1);
        detector->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptors_2);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < 0.75f * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        if (good_matches.size() < 15)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough matches");
            return;
        }

        std::vector<cv::Point2f> points1, points2;
        for (size_t i = 0; i < good_matches.size(); i++)
        {
            points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        }

        homography_ = cv::findHomography(points1, points2, cv::RANSAC);
    }

    cv::Mat homography_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageStitcher>());
    rclcpp::shutdown();
    return 0;
}