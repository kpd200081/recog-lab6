#include "polfit.h"

#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

#define CUTED_SIZE 640

static int x_offset(490), y_offset(0), w1(855), w2(220), h(175), w_size_t(32), w_size(64),
    treshold(200);

void wtrb_pos(int pos, void*) {
    if(pos < 1) {
        w_size_t = 1;
        setTrackbarPos("window size", "Main", 1);
    }
    if(CUTED_SIZE % w_size_t == 0) {
        w_size = w_size_t;
    }
    setTrackbarPos("window size", "Main", w_size);
}

std::vector<Point2f> pointsToDrawLine(Mat& img, std::vector<float>& line) {
    double theMult = max(img.size[0], img.size[1]);
    // calculate start point
    Point startPoint;
    startPoint.x = line[2] - theMult * line[0]; // x0
    startPoint.y = line[3] - theMult * line[1]; // y0
    // calculate end point
    Point endPoint;
    endPoint.x = line[2] + theMult * line[0]; // x[1]
    endPoint.y = line[3] + theMult * line[1]; // y[1]

    // draw overlay of bottom lines on image
    clipLine(Size(img.size[0], img.size[1]), startPoint, endPoint);
    std::vector<Point2f> ret;
    ret.push_back(startPoint);
    ret.push_back(endPoint);
    return ret;
    //    cv::line(img, startPoint, endPoint, color, thickness, 8, 0);
}

int main() {
    VideoCapture cap("../solidWhiteRight.mp4");

    Mat frame, res, transform, hls, dtc, dtc_show, window, final;

    while(frame.empty()) {
        cap >> frame;
    }

    namedWindow("Main", WINDOW_AUTOSIZE);
    namedWindow("Cutted", WINDOW_AUTOSIZE);
    namedWindow("Detected", WINDOW_AUTOSIZE);
    namedWindow("Final", WINDOW_AUTOSIZE);
    createTrackbar("x offset", "Main", &x_offset, frame.size[1]);
    createTrackbar("y offset", "Main", &y_offset, frame.size[0]);
    createTrackbar("down width", "Main", &w1, frame.size[1]);
    createTrackbar("up width", "Main", &w2, frame.size[1]);
    createTrackbar("height", "Main", &h, frame.size[0]);
    createTrackbar("window size", "Main", &w_size_t, CUTED_SIZE / 4, &wtrb_pos);
    setTrackbarMin("window size", "Main", 4);
    createTrackbar("white treshold in window", "Main", &treshold, 3000);

    std::vector<std::vector<Point>> pols;
    std::vector<Point> pol;
    std::vector<Point2f> persp_dst, persp_src;
    persp_dst.push_back(Point2f(0, 0));
    persp_dst.push_back(Point2f(CUTED_SIZE, 0));
    persp_dst.push_back(Point2f(CUTED_SIZE, CUTED_SIZE));
    persp_dst.push_back(Point2f(0, CUTED_SIZE));

    std::vector<float> coefs_left, coefs_right;
    while(true) {
        cap >> frame;

        if(frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        frame.copyTo(res);
        frame.copyTo(final);

        pol.clear();
        pols.clear();
        pol.push_back(Point(x_offset - w1 / 2, frame.size[0] - y_offset));
        pol.push_back(Point(x_offset + w1 / 2, frame.size[0] - y_offset));
        pol.push_back(Point(x_offset + w2 / 2, frame.size[0] - y_offset - h));
        pol.push_back(Point(x_offset - w2 / 2, frame.size[0] - y_offset - h));
        pols.push_back(pol);
        polylines(frame, pols, true, Scalar(0, 255, 0), 2);

        putText(
            frame,
            format("FPS: %3.2f", cap.get(CAP_PROP_FPS)),
            Point(10, 25),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        putText(
            frame,
            format("Frame:        %5.0f", cap.get(CAP_PROP_POS_FRAMES)),
            Point(10, 50),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        putText(
            frame,
            format("Frames total: %5.0f", cap.get(CAP_PROP_FRAME_COUNT)),
            Point(10, 75),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        persp_src.clear();
        persp_src.push_back(Point2f(x_offset - w2 / 2, frame.size[0] - y_offset - h));
        persp_src.push_back(Point2f(x_offset + w2 / 2, frame.size[0] - y_offset - h));
        persp_src.push_back(Point2f(x_offset + w1 / 2, frame.size[0] - y_offset));
        persp_src.push_back(Point2f(x_offset - w1 / 2, frame.size[0] - y_offset));
        transform = getPerspectiveTransform(persp_src, persp_dst);
        warpPerspective(res, res, transform, { int(persp_dst[2].x), int(persp_dst[2].y) });

        imshow("Cutted", res);

        cvtColor(res, hls, COLOR_BGR2HLS);
        inRange(hls, Scalar(0, 0, 0), Scalar(255, 215, 255), dtc);
        bitwise_not(dtc, dtc);
        std::vector<Point2f> dtc_left;
        std::vector<Point2f> dtc_right;
        dtc.copyTo(dtc_show);
        bool has_far_left = false;
        bool has_far_right = false;
        for(int i = 0; i < CUTED_SIZE / w_size; i++) {
            for(int j = 0; j < 2 * CUTED_SIZE / w_size - 1; j++) {
                Rect roi(j * w_size / 2, i * w_size, w_size, w_size);
                window = dtc(roi);
                Moments ms = moments(window, true);
                if(ms.m00 > treshold) {
                    Point2f p(
                        j * w_size / 2 + float(ms.m10 / ms.m00),
                        i* w_size + float(ms.m01 / ms.m00));
                    if(j < CUTED_SIZE / w_size) {
                        bool exist = false;
                        for(size_t k = 0; k < dtc_left.size(); k++) {
                            if(norm(dtc_left[k] - p) < 10) {
                                exist = true;
                            }
                            if(norm(dtc_left[k] - p) > CUTED_SIZE / 4) {
                                has_far_left = true;
                            }
                        }
                        if(!exist) {
                            dtc_left.push_back(p);
                            circle(dtc_show, p, 5, Scalar(128), -1);
                        }
                    } else {
                        bool exist = false;
                        for(size_t k = 0; k < dtc_right.size(); k++) {
                            if(norm(dtc_right[k] - p) < 10) {
                                exist = true;
                            }
                            if(norm(dtc_right[k] - p) > CUTED_SIZE / 4) {
                                has_far_right = true;
                            }
                        }
                        if(!exist) {
                            dtc_right.push_back(p);
                            circle(dtc_show, p, 5, Scalar(128), -1);
                        }
                    }
                    rectangle(dtc_show, roi, Scalar(255), 2);
                }
            }
        }

        //        std::vector<float> left_line;
        //        fitLine(dtc_left, left_line, DIST_L2, 0, 0.01, 0.01);
        //        std::vector<Point2f> left = pointsToDrawLine(dtc_show, left_line);
        //        line(dtc_show, left[0], left[1], Scalar(128), 5);

        //        std::vector<float> right_line;
        //        fitLine(dtc_right, right_line, DIST_L2, 0, 0.01, 0.01);
        //        std::vector<Point2f> right = pointsToDrawLine(dtc_show, right_line);
        //        line(dtc_show, right[0], right[1], Scalar(128), 5);
        if(has_far_left) {
            std::vector<float> xs;
            std::vector<float> ys;
            for(Point2f p: dtc_left) {
                xs.push_back(p.x);
                ys.push_back(p.y);
            }
            PolynomialRegression<float> r_left;
            r_left.fitIt(ys, xs, 2, coefs_left);
        }
        std::vector<Point2f> left;
        for(float y = 0; y < dtc_show.rows; y += 0.1) {
            left.push_back(
                Point2f(coefs_left[0] + coefs_left[1] * y + coefs_left[2] * pow(y, 2), y));
        }
        for(Point2f p: left) {
            circle(dtc_show, p, 2, Scalar(128));
        }

        if(has_far_right) {
            std::vector<float> xs;
            std::vector<float> ys;
            for(Point2f p: dtc_right) {
                xs.push_back(p.x);
                ys.push_back(p.y);
            }
            PolynomialRegression<float> r_right;
            r_right.fitIt(ys, xs, 2, coefs_right);
        }
        std::vector<Point2f> right;
        for(float y = 0; y < dtc_show.rows; y += 0.1) {
            right.push_back(
                Point2f(coefs_right[0] + coefs_right[1] * y + coefs_right[2] * pow(y, 2), y));
        }
        for(Point2f p: right) {
            circle(dtc_show, p, 2, Scalar(128));
        }

        imshow("Detected", dtc_show);

        std::vector<Point2f> left_f;
        std::vector<Point2f> right_f;
        perspectiveTransform(left, left_f, transform.inv());
        perspectiveTransform(right, right_f, transform.inv());
        for(Point2f p: left_f) {
            circle(frame, p, 2, Scalar(0, 0, 255));
        }
        for(Point2f p: right_f) {
            circle(frame, p, 2, Scalar(0, 0, 255));
        }

        float r_left = pow(1 + pow(2 * coefs_left[2] * dtc_show.rows + coefs_left[1], 2), 1.5) /
            std::abs(2 * coefs_left[2]);
        float r_right = pow(1 + pow(2 * coefs_right[2] * dtc_show.rows + coefs_right[1], 2), 1.5) /
            std::abs(2 * coefs_right[2]);

        putText(
            frame,
            format("Turn radius(in pixels): %.1f", (r_left + r_right) / 2.0),
            Point(10, 100),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        imshow("Main", frame);

        std::vector<Point> poligon;
        for(Point2f p: left_f) {
            poligon.push_back(p);
        }
        std::reverse(right_f.begin(), right_f.end());
        for(Point2f p: right_f) {
            poligon.push_back(p);
        }

        Mat layer = Mat::zeros(final.size(), CV_8UC3);
        fillPoly(layer, poligon, Scalar(0, 250, 0));
        addWeighted(final, 1, layer, 0.3, 0, final);

        imshow("Final", final);

        if(waitKey(10) == 27)
            break; // ESC
    }

    cap.release();
    return 0;
}
