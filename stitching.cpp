#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

bool is_image(const fs::path &p) {
    const vector<string> exts = {".jpg",".jpeg",".png",".bmp",".tiff"};
    string ext = p.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(),
              [](unsigned char c){ return tolower(c); });
    return find(exts.begin(), exts.end(), ext) != exts.end();
}

vector<fs::path> get_all_images(const fs::path& dir) {
    vector<fs::path> images;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && is_image(entry.path())) {
                images.push_back(entry.path());
            }
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
    }
    sort(images.begin(), images.end());
    return images;
}

int main() {
    string directory = string(DATA_PATH) + "/set1";
    vector<fs::path> image_paths = get_all_images(directory);

    if (image_paths.size() < 2) {
        cerr << "Need at least 2 images." << endl;
        return -1;
    }

    vector<Mat> images;
    for (auto &p : image_paths) {
        Mat img = imread(p.string());
        if (img.empty()) {
            cerr << "Failed to load: " << p << endl;
            return -1;
        }
        images.push_back(img);
    }

    vector<Mat> gray(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        cvtColor(images[i], gray[i], COLOR_BGR2GRAY);
    }

    Ptr<ORB> orb = ORB::create(4000);
    vector<vector<KeyPoint>> keypoints(images.size());
    vector<Mat> descriptors(images.size());

    for (size_t i = 0; i < images.size(); i++) {
        orb->detectAndCompute(gray[i], noArray(), keypoints[i], descriptors[i]);
    }

    Ptr<DescriptorMatcher> matcher =
        DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    vector<vector<vector<DMatch>>> matches(images.size() - 1);

    for (size_t i = 0; i < images.size() - 1; i++) {
        if (descriptors[i].empty() || descriptors[i+1].empty()) {
            cerr << "Missing descriptors at pair " << i << endl;
            return -1;
        }

        Mat d1, d2;
        descriptors[i].convertTo(d1, CV_32F);
        descriptors[i+1].convertTo(d2, CV_32F);

        matcher->knnMatch(d1, d2, matches[i], 2);
    }

    vector<vector<DMatch>> good_matches(images.size() - 1);

    for (size_t i = 0; i < matches.size(); i++) {
        for (auto &m : matches[i]) {
            if (m.size() < 2) continue;

            if (m[0].distance < 0.67 * m[1].distance) {
                good_matches[i].push_back(m[0]);
            }
        }

        cout << "Pair " << i << " matches: "
             << good_matches[i].size() << endl;

        if (good_matches[i].size() < 10) {
            cerr << "Not enough good matches at pair " << i << endl;
            return -1;
        }
    }

    vector<Mat> Hs(images.size() - 1);

    for (size_t i = 0; i < images.size() - 1; i++) {
        vector<Point2f> pts1, pts2;

        for (auto &m : good_matches[i]) {
            pts1.push_back(keypoints[i][m.queryIdx].pt);
            pts2.push_back(keypoints[i+1][m.trainIdx].pt);
        }

        Mat mask;
        Hs[i] = findHomography(pts2, pts1, RANSAC, 3.0, mask);

        int inliers = countNonZero(mask);
        cout << "Pair " << i << " inliers: " << inliers << endl;

        if (Hs[i].empty() || inliers < 10) {
            cerr << "Bad homography at pair " << i << endl;
            return -1;
        }
    }

    int center = images.size() / 2;
    vector<Mat> H_global(images.size());
    H_global[center] = Mat::eye(3,3,CV_64F);

    for (int i = center + 1; i < images.size(); i++) {
        H_global[i] = Hs[i-1] * H_global[i-1];
    }

    for (int i = center - 1; i >= 0; i--) {
        H_global[i] = H_global[i+1] * Hs[i].inv();
    }

    float min_x = 0, min_y = 0;
    float max_x = images[0].cols;
    float max_y = images[0].rows;

    for (size_t i = 0; i < images.size(); i++) {
        vector<Point2f> corners = {
            {0,0},
            {(float)images[i].cols,0},
            {(float)images[i].cols,(float)images[i].rows},
            {0,(float)images[i].rows}
        };

        vector<Point2f> transformed;
        perspectiveTransform(corners, transformed, H_global[i]);

        for (auto &p : transformed) {
            min_x = min(min_x, p.x);
            min_y = min(min_y, p.y);
            max_x = max(max_x, p.x);
            max_y = max(max_y, p.y);
        }
    }

    int width  = (int)ceil(max_x - min_x);
    int height = (int)ceil(max_y - min_y);

    Mat T = (Mat_<double>(3,3) <<
        1, 0, -min_x,
        0, 1, -min_y,
        0, 0, 1);

    Mat result = Mat::zeros(Size(width, height), images[0].type());

    for (size_t i = 0; i < images.size(); i++) {
        Mat warped;
        warpPerspective(images[i], warped,
                        T * H_global[i],
                        result.size(),
                        INTER_LINEAR,
                        BORDER_TRANSPARENT);

        Mat grayWarped, mask;
        cvtColor(warped, grayWarped, COLOR_BGR2GRAY);
        threshold(grayWarped, mask, 0, 255, THRESH_BINARY);

        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (mask.at<uchar>(y,x)) {
                    Vec3b w = warped.at<Vec3b>(y,x);
                    Vec3b &r = result.at<Vec3b>(y,x);

                    if (r == Vec3b(0,0,0))
                        r = w;
                    else
                        r = (r + w) / 2;
                }
            }
        }
    }

    if (imwrite("stitched_result1.jpg", result))
        cout << "Panorama saved successfully!" << endl;
    else
        cerr << "Failed to save panorama." << endl;

    return 0;
}