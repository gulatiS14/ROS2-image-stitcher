#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <filesystem>
#include <iostream>

class ImageStitcher : public rclcpp::Node
{
public:
    ImageStitcher() : Node("image_stitcher")
    {
        left_sub_.subscribe(this, "/left_camera/image_raw");
        right_sub_.subscribe(this, "/right_camera/image_raw");

        sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), left_sub_, right_sub_);
        sync_->registerCallback(std::bind(&ImageStitcher::image_callback, this, std::placeholders::_1, std::placeholders::_2));

        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ImageStitcher::timer_callback, this));
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
    {
        try {
            cv_bridge::CvImagePtr left_cv_ptr = cv_bridge::toCvCopy(left_msg, sensor_msgs::image_encodings::BGR8);
            cv_bridge::CvImagePtr right_cv_ptr = cv_bridge::toCvCopy(right_msg, sensor_msgs::image_encodings::BGR8);

            left_img_ = left_cv_ptr->image;
            right_img_ = right_cv_ptr->image;

            RCLCPP_INFO(this->get_logger(), "Received synchronized images");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void timer_callback()
    {
        if (left_img_.empty() || right_img_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No images received yet");
            return;
        }

        cv::Mat result = stitch(left_img_, right_img_);

        if (!result.empty())
        {
            cv::imwrite("/app/stitched_output.jpg", result);
            RCLCPP_INFO(this->get_logger(), "Stitched image saved as 'stitched_output.jpg'");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to stitch images");
        }
    }

    cv::Mat stitch(const cv::Mat& left_img, const cv::Mat& right_img)
    {
        cv::Mat left_gray, right_gray;
        cv::cvtColor(left_img, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_img, right_gray, cv::COLOR_BGR2GRAY);

        auto detector = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;

        detector->detectAndCompute(left_gray, cv::Mat(), keypoints_1, descriptors_1);
        detector->detectAndCompute(right_gray, cv::Mat(), keypoints_2, descriptors_2);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < 0.7f * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        if (good_matches.size() < 10)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough matches");
            return cv::Mat();
        }

        std::vector<cv::Point2f> src_pts, dst_pts;
        for (size_t i = 0; i < good_matches.size(); i++)
        {
            src_pts.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            dst_pts.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        }

        cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC);

        int h1 = left_img.rows, w1 = left_img.cols;
        int h2 = right_img.rows, w2 = right_img.cols;

        std::vector<cv::Point2f> pts = {
            cv::Point2f(0, 0), cv::Point2f(0, h1),
            cv::Point2f(w1, h1), cv::Point2f(w1, 0)
        };
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(pts, dst, H);

        pts.insert(pts.end(), {cv::Point2f(0, 0), cv::Point2f(0, h2),
                               cv::Point2f(w2, h2), cv::Point2f(w2, 0)});
        dst.insert(dst.end(), {cv::Point2f(0, 0), cv::Point2f(0, h2),
                               cv::Point2f(w2, h2), cv::Point2f(w2, 0)});

        cv::Rect bounds = cv::boundingRect(dst);
        cv::Mat translation = (cv::Mat_<double>(3, 3) << 
            1, 0, -bounds.x,
            0, 1, -bounds.y,
            0, 0, 1
        );

        cv::Mat result;
        cv::warpPerspective(left_img, result, translation * H, bounds.size());
        cv::Mat roi(result, cv::Rect(-bounds.x, -bounds.y, right_img.cols, right_img.rows));
        right_img.copyTo(roi);

        return result;
    }

    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
    std::shared_ptr<Synchronizer> sync_;

    rclcpp::TimerBase::SharedPtr timer_;
    cv::Mat left_img_, right_img_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageStitcher>());
    rclcpp::shutdown();
    return 0;
}