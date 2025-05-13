#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace chrono;

const int NUM_THREADS = 4;
const double ROI_FACTOR = 0.5;
const double PI = 3.14159265;

struct Metrics {
    double sobelTime = 0;
    double houghTime = 0;
    double totalTime = 0;
    int numLinesDetected = 0;
    double averageLineLength = 0;
    double processingFPS = 0;
};

mutex acc_mutex;

struct ThreadParams {
    Mat* input;
    Mat* output;
    int startRow;
    int endRow;
    int threshold;
};

void sobelThread(ThreadParams params) {
    for(int y = params.startRow; y < params.endRow; y++) {
        for(int x = 1; x < params.input->cols - 1; x++) {
            float gx = -params.input->at<uchar>(y-1, x-1) - 2*params.input->at<uchar>(y, x-1) - params.input->at<uchar>(y+1, x-1) +
                      params.input->at<uchar>(y-1, x+1) + 2*params.input->at<uchar>(y, x+1) + params.input->at<uchar>(y+1, x+1);
            float gy = -params.input->at<uchar>(y-1, x-1) - 2*params.input->at<uchar>(y-1, x) - params.input->at<uchar>(y-1, x+1) +
                      params.input->at<uchar>(y+1, x-1) + 2*params.input->at<uchar>(y+1, x) + params.input->at<uchar>(y+1, x+1);
            float magnitude = sqrt(gx*gx + gy*gy) / 4.0;
            params.output->at<uchar>(y, x) = magnitude > params.threshold ? magnitude : 0;
        }
    }
}

Mat applySobel(Mat& input, int threshold = 50) {
    Mat output = Mat::zeros(input.size(), CV_8UC1);
    int rowsPerThread = input.rows / NUM_THREADS;
    vector<thread> threads;
    for(int i = 0; i < NUM_THREADS; i++) {
        ThreadParams params = {
            &input,
            &output,
            i * rowsPerThread,
            (i == NUM_THREADS-1) ? input.rows : (i+1) * rowsPerThread,
            threshold
        };
        threads.emplace_back(sobelThread, params);
    }
    for(auto& t : threads) t.join();
    return output;
}

int applyHoughLinesP_adaptatif(Mat& edges, Mat& result, int roi_y, int roi_mode, Metrics& metrics, bool& is_straight, bool permissif=false) {
    vector<Vec4i> linesP;
    int threshold = permissif ? 10 : 25;
    int minLineLength = permissif ? 15 : 30;
    int maxLineGap = permissif ? 50 : 30;
    HoughLinesP(edges, linesP, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
    int numLines = 0;
    double totalLength = 0;
    int img_center = result.cols / 2;
    int nb_verticales = 0;
    int nb_centrales = 0;
    for(const auto& l : linesP) {
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
        int x1 = l[0], x2 = l[2];
        int x_center = (x1 + x2) / 2;
        bool is_center = abs(x_center - img_center) < img_center * (roi_mode ? 0.7 : 0.5);
        bool is_vertical = (abs(angle) < 30 || abs(angle) > 150);
        if ((is_vertical || permissif) && length > (permissif ? 10 : 40) && is_center) {
            Point pt1(l[0], l[1] + roi_y);
            Point pt2(l[2], l[3] + roi_y);
            line(result, pt1, pt2, Scalar(0, 255, 0), 3, LINE_AA);
            numLines++;
            totalLength += length;
            if(is_vertical) nb_verticales++;
            if(is_center) nb_centrales++;
        }
    }
    is_straight = (numLines > 1 && nb_verticales > 0.7 * numLines && nb_centrales > 0.7 * numLines);
    metrics.numLinesDetected = numLines;
    metrics.averageLineLength = numLines > 0 ? totalLength / numLines : 0;
    return numLines;
}

int main() {
    VideoCapture cap("nD_1.mp4");
    if(!cap.isOpened()) {
        cerr << "Erreur : impossible d'ouvrir la vidéo simulation_virage.mp4" << endl;
        return 1;
    }
    int frame_count = 0;
    double total_fps = 0;
    double total_lines = 0;
    double total_length = 0;
    auto global_start = high_resolution_clock::now();
    
    // Pour sauvegarder la vidéo traitée (optionnel)
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    VideoWriter writer("result_simulation_virage.mp4", VideoWriter::fourcc('a','v','c','1'), fps, Size(w, h));
    
    while(true) {
        Mat frame;
        if(!cap.read(frame)) break;
        Metrics metrics;
        auto start = high_resolution_clock::now();
        // --- Pipeline identique à image.cpp ---
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 1.5);
        Mat hsv, mask_yellow;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Scalar lower_yellow(15, 50, 50), upper_yellow(40, 255, 255);
        inRange(hsv, lower_yellow, upper_yellow, mask_yellow);
        int roi_mode = 0;
        int roi_y = gray.rows * 0.4;
        int roi_height = gray.rows - roi_y;
        Rect roi_rect(0, roi_y, gray.cols, roi_height);
        Mat roi_gray = gray(roi_rect);
        Mat roi_mask = mask_yellow(roi_rect);
        Mat edges = applySobel(roi_gray, 25);
        bitwise_and(edges, roi_mask, edges);
        Mat result = frame.clone();
        bool is_straight = false;
        int numLines = applyHoughLinesP_adaptatif(edges, result, roi_y, roi_mode, metrics, is_straight);
        if(numLines < 2) {
            roi_mode = 1;
            roi_y = gray.rows / 3;
            roi_height = gray.rows - roi_y;
            roi_rect = Rect(0, roi_y, gray.cols, roi_height);
            roi_gray = gray(roi_rect);
            roi_mask = mask_yellow(roi_rect);
            edges = applySobel(roi_gray, 20);
            bitwise_and(edges, roi_mask, edges);
            result = frame.clone();
            is_straight = false;
            numLines = applyHoughLinesP_adaptatif(edges, result, roi_y, roi_mode, metrics, is_straight, true);
        }
        auto end = high_resolution_clock::now();
        metrics.totalTime = duration_cast<milliseconds>(end - start).count();
        metrics.processingFPS = 1000.0 / metrics.totalTime;
        // --- Fin pipeline ---
        total_fps += metrics.processingFPS;
        total_lines += metrics.numLinesDetected;
        total_length += metrics.averageLineLength;
        frame_count++;
        writer.write(result);
    }
    auto global_end = high_resolution_clock::now();
    double global_time = duration_cast<milliseconds>(global_end - global_start).count() / 1000.0;
    cout << fixed << setprecision(2);
    cout << "\n=== Métriques globales vidéo ===" << endl;
    cout << "Nombre de frames traitées : " << frame_count << endl;
    cout << "FPS moyen : " << (frame_count / global_time) << endl;
    cout << "Nombre moyen de lignes détectées : " << (total_lines / frame_count) << endl;
    cout << "Longueur moyenne des lignes : " << (total_length / frame_count) << " pixels" << endl;
    cout << "Temps total de traitement : " << global_time << " s" << endl;
    cout << "===============================\n" << endl;
    cap.release();
    writer.release();
    return 0;
} 