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

// Constantes
const int NUM_THREADS = 4; // demain tester avec 3 4 5 threads
const double ROI_FACTOR = 0.5;
const int THETA_STEP = 4;
const double PI = 3.14159265;

// Structure pour les métriques
struct Metrics {
    double sobelTime;
    double houghTime;
    double totalTime;
    int numLinesDetected;
    double averageLineLength;
    double processingFPS;
};

// Mutex pour l'accès concurrent
mutex acc_mutex;

// Structure pour les paramètres des threads
struct ThreadParams {
    Mat* input;
    Mat* output;
    int startRow;
    int endRow;
    int threshold;
};

// Fonction optimisée de Sobel
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

// Fonction optimisée de Hough
void houghThread(Mat& edges, Mat& accumulator, int startRow, int endRow) {
    for(int y = startRow; y < endRow; y++) {
        for(int x = 0; x < edges.cols; x++) {
            if(edges.at<uchar>(y, x) > 0) {
                for(int theta = 0; theta < 180; theta += THETA_STEP) {
                    double rho = x * cos(theta * PI / 180.0) + y * sin(theta * PI / 180.0);
                    int rhoIndex = cvRound(rho);
                    
                    if(rhoIndex >= 0 && rhoIndex < accumulator.rows) {
                        lock_guard<mutex> lock(acc_mutex);
                        accumulator.at<ushort>(rhoIndex, theta/THETA_STEP)++;
                    }
                }
            }
        }
    }
}

// Classe principale pour le traitement d'image
class RoadDetector {
private:
    Metrics metrics;
    vector<thread> threads;
    
public:
    RoadDetector() {
        resetMetrics();
    }
    
    void resetMetrics() {
        metrics.sobelTime = 0;
        metrics.houghTime = 0;
        metrics.totalTime = 0;
        metrics.numLinesDetected = 0;
        metrics.averageLineLength = 0;
        metrics.processingFPS = 0;
    }
    
    Mat processImage(const string& imagePath) {
        auto startTotal = high_resolution_clock::now();
        
        // Chargement de l'image
        Mat input = imread(imagePath);
        if(input.empty()) {
            throw runtime_error("Impossible de charger l'image: " + imagePath);
        }
        
        // Conversion en niveaux de gris
        Mat gray;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        
        // Flou gaussien pour réduire le bruit
        GaussianBlur(gray, gray, Size(3, 3), 1.5);
        
        // Masque couleur jaune (en HSV) - plage élargie
        Mat hsv, mask_yellow;
        cvtColor(input, hsv, COLOR_BGR2HSV);
        // Plage élargie pour le jaune
        Scalar lower_yellow(10, 60, 60), upper_yellow(45, 255, 255);
        inRange(hsv, lower_yellow, upper_yellow, mask_yellow);
        
        // ROI dynamique : par défaut moitié inférieure, sinon 2/3 si route droite
        int roi_mode = 0; // 0 = normal, 1 = spécial route droite
        int roi_y = gray.rows * ROI_FACTOR;
        int roi_height = gray.rows - roi_y;
        // On commence avec la moitié inférieure
        Rect roi_rect(0, roi_y, gray.cols, roi_height);
        Mat roi_gray = gray(roi_rect);
        Mat roi_mask = mask_yellow(roi_rect);
        
        // Application du filtre de Sobel sur la zone jaune uniquement
        auto startSobel = high_resolution_clock::now();
        Mat edges = applySobel(roi_gray, 30);
        bitwise_and(edges, roi_mask, edges);
        auto endSobel = high_resolution_clock::now();
        metrics.sobelTime = duration_cast<milliseconds>(endSobel - startSobel).count();
        
        // Application de la transformée de Hough améliorée
        auto startHough = high_resolution_clock::now();
        Mat result = input.clone();
        bool is_straight = false;
        int numLines = applyHoughLinesP_adaptatif(edges, result, roi_y, roi_mode, is_straight);
        // Si peu de lignes détectées, on relance avec des paramètres plus permissifs et une ROI plus grande
        if(numLines < 2) {
            // On prend les 2/3 inférieurs
            roi_mode = 1;
            roi_y = gray.rows / 3;
            roi_height = gray.rows - roi_y;
            roi_rect = Rect(0, roi_y, gray.cols, roi_height);
            roi_gray = gray(roi_rect);
            roi_mask = mask_yellow(roi_rect);
            edges = applySobel(roi_gray, 20);
            bitwise_and(edges, roi_mask, edges);
            result = input.clone();
            is_straight = false;
            numLines = applyHoughLinesP_adaptatif(edges, result, roi_y, roi_mode, is_straight, true);
        }
        auto endHough = high_resolution_clock::now();
        metrics.houghTime = duration_cast<milliseconds>(endHough - startHough).count();
        
        // Calcul des métriques finales
        auto endTotal = high_resolution_clock::now();
        metrics.totalTime = duration_cast<milliseconds>(endTotal - startTotal).count();
        metrics.processingFPS = 1000.0 / metrics.totalTime;
        
        return result;
    }
    
    // Sobel avec seuil paramétrable
    Mat applySobel(Mat& input, int threshold = 50) {
        Mat output = Mat::zeros(input.size(), CV_8UC1);
        int rowsPerThread = input.rows / NUM_THREADS;
        
        threads.clear();
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
        
        for(auto& t : threads) {
            t.join();
        }
        
        return output;
    }
    
    // HoughLinesP adaptative avec détection de route droite
    int applyHoughLinesP_adaptatif(Mat& edges, Mat& result, int roi_y, int roi_mode, bool& is_straight, bool permissif=false) {
        vector<Vec4i> linesP;
        // Paramètres adaptatifs
        int threshold = permissif ? 15 : 30;
        int minLineLength = permissif ? 20 : 40;
        int maxLineGap = permissif ? 40 : 20;
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
            bool is_center = abs(x_center - img_center) < img_center * (roi_mode ? 0.6 : 0.45);
            bool is_vertical = (abs(angle) < 25 || abs(angle) > 155);
            // On garde plus large si mode permissif
            if ((is_vertical || permissif) && length > (permissif ? 15 : 50) && is_center) {
                Point pt1(l[0], l[1] + roi_y);
                Point pt2(l[2], l[3] + roi_y);
                line(result, pt1, pt2, Scalar(0, 255, 0), 3, LINE_AA);
                numLines++;
                totalLength += length;
                if(is_vertical) nb_verticales++;
                if(is_center) nb_centrales++;
            }
        }
        // Détection automatique de route droite : majorité de lignes verticales et centrales
        is_straight = (numLines > 1 && nb_verticales > 0.7 * numLines && nb_centrales > 0.7 * numLines);
        metrics.numLinesDetected = numLines;
        metrics.averageLineLength = numLines > 0 ? totalLength / numLines : 0;
        return numLines;
    }
    
    void printMetrics() {
        cout << "\n=== Métriques de Performance ===" << endl;
        cout << "Temps de traitement Sobel: " << fixed << setprecision(2) << metrics.sobelTime << " ms" << endl;
        cout << "Temps de traitement Hough: " << metrics.houghTime << " ms" << endl;
        cout << "Temps total de traitement: " << metrics.totalTime << " ms" << endl;
        cout << "FPS: " << metrics.processingFPS << endl;
        cout << "Nombre de lignes détectées: " << metrics.numLinesDetected << endl;
        cout << "Longueur moyenne des lignes: " << metrics.averageLineLength << " pixels" << endl;
        cout << "==============================\n" << endl;
    }
};

int main() {
    try {
        RoadDetector detector;
        
        // Traitement de la première image
        cout << "Traitement de route_low.jpg..." << endl;
        Mat result1 = detector.processImage("route_low.jpg");
        detector.printMetrics();
        imwrite("result_route_low.jpg", result1);
        
        // Réinitialisation des métriques
        detector.resetMetrics();
        
        // Traitement de la deuxième image
        cout << "Traitement de route_virage.jpg..." << endl;
        Mat result2 = detector.processImage("route_virage.jpg");
        detector.printMetrics();
        imwrite("result_route_virage.jpg", result2);
        
    } catch(const exception& e) {
        cerr << "Erreur: " << e.what() << endl;
        return 1;
    }
    
    return 0;
} 