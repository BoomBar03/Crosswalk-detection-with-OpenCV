#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "common.h"
#include <vector>
#include <iostream>
#include <random>
#include <queue>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

using namespace cv;
using namespace std;

void testOpenImage() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src;
		src = imread(fname, IMREAD_COLOR); // Ensure the image is opened in color
		imshow("image", src);
		waitKey();
	}
}

typedef struct {
	int minX, minY, maxX, maxY, area;
} LabeledObject;


Mat douaTreceri(Mat img, int* labelSize)
{
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
	Mat labels = Mat::zeros(img.size(), CV_32SC1);

	int label = 0;
	int height = img.rows;
	int width = img.cols;

	// Vectorii de directie 8-neighborhood
	int di[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dj[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	// Prima trecere: asigneaza initial etichetele si inregistreaza echivalentele
	vector<vector<int>> edges(5000);
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (img.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
				vector<int> L;

				for (int d = 0; d < 8; d++) {
					if (labels.at<int>(i + di[d], j + dj[d]) > 0) {
						L.push_back(labels.at<int>(i + di[d], j + dj[d]));
					}
				}

				if (L.empty()) {
					label++;
					labels.at<int>(i, j) = label;
				}
				else {
					int x = *min_element(L.begin(), L.end());
					labels.at<int>(i, j) = x;

					for (int y : L) {
						if (y != x) {
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}
			}
		}
	}

	// A doua trecere: rezolva echivalenta utilizand BFS
	vector<int> newLabels(label + 1, 0);
	int newLabel = 0;

	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			queue<int> q;
			newLabels[i] = newLabel;
			q.push(i);

			while (!q.empty()) {
				int x = q.front();
				q.pop();

				for (int y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						q.push(y);
					}
				}
			}
		}
	}

	// Reasigneaza etichetele bazate pe noua etichetare
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
		}
	}

	*labelSize = newLabel;

	
	vector<Vec3b> colors(newLabel + 1);
	default_random_engine eng;
	uniform_int_distribution<int> distrib(0, 255);
	for (int i = 0; i <= newLabel; i++) {
		colors[i] = Vec3b(distrib(eng), distrib(eng), distrib(eng));
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels.at<int>(i, j) == 0) {
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else {
				dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
			}
		}
	}

	return labels;
}

Mat median_filter1(const Mat& src) {
	int width = 5;
	Mat dst(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			std::vector<int> values;

			for (int ii = 0; ii < width; ii++) {
				for (int jj = 0; jj < width; jj++) {
					int ipixel = i + ii - width / 2;
					int jpixel = j + jj - width / 2;

					if (ipixel >= 0 && ipixel < src.rows && jpixel >= 0 && jpixel < src.cols) {
						values.push_back(src.at<uchar>(ipixel, jpixel));
					}
				}
			}

			// Calculate median
			sort(values.begin(), values.end());
			int median;
			if (values.size() % 2 == 0) {
				median = (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
			}
			else {
				median = values[values.size() / 2];
			}

			dst.at<uchar>(i, j) = median;
		}
	}

	return dst;
}

LabeledObject calculateObjectProperties(const Mat& labels, int label) {
	int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN, area = 0;

	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			if (labels.at<int>(i, j) == label) {
				if (j < minX) minX = j;
				if (i < minY) minY = i;
				if (j > maxX) maxX = j;
				if (i > maxY) maxY = i;
				area++;
			}
		}
	}

	LabeledObject obj = { minX, minY, maxX, maxY, area };
	return obj;
}

Mat detectZebraCrossing(const Mat& binarizedImage, Mat& originalImage) {
	// Etichetarea componentelor conectate folosind douaTreceri
	Mat labels;
	int nLabels;
	labels = douaTreceri(binarizedImage, &nLabels);

	Mat contourOutput = originalImage.clone();
	std::vector<Rect> detectedRectangles;

	// Verificarea fiecarei componente
	for (int label = 1; label <= nLabels; ++label) {
		LabeledObject obj = calculateObjectProperties(labels, label);

		// Filtrarea componentelor mici s a celor care nu sunt dreptunghiuri
		int width = obj.maxX - obj.minX + 1;
		int height = obj.maxY - obj.minY + 1;
		if (obj.area > 350 && width > height && (width> height || height> width )) {
			Rect boundingBox(obj.minX, obj.minY, width, height);
			detectedRectangles.push_back(boundingBox);
		}
	}

	// Gruparea dreptunghiurilor consecutive
	std::vector<std::vector<Rect>> groupedRectangles;
	std::vector<Rect> currentGroup;

	for (size_t i = 0; i < detectedRectangles.size(); ++i) {
		if (currentGroup.empty()) {
			currentGroup.push_back(detectedRectangles[i]);
		}
		else {
			Rect lastRect = currentGroup.back();
			Rect currentRect = detectedRectangles[i];

			if (abs(currentRect.x - lastRect.x) < 80 && abs(currentRect.y - lastRect.y) < 80) {
				currentGroup.push_back(currentRect);
			}
			else {
				if (currentGroup.size() > 2) {
					groupedRectangles.push_back(currentGroup);
				}
				currentGroup.clear();
				currentGroup.push_back(currentRect);
			}
		}
	}
	if (currentGroup.size() > 2) {
		groupedRectangles.push_back(currentGroup);
	}

	// Desenarea dreptunghiurilor grupate pe imaginea de iesire
	for (const auto& group : groupedRectangles) {
		std::vector<Point> groupPoints;
		for (const auto& rect : group) {
			for (int y = rect.y; y < rect.y + rect.height; ++y) {
				for (int x = rect.x; x < rect.x + rect.width; ++x) {
					groupPoints.push_back(Point(x, y));
				}
			}
		}

		if (!groupPoints.empty()) {


			// Desenarea unui dreptunghi în jurul grupului
			Rect boundingBox = boundingRect(Mat(groupPoints));
			rectangle(contourOutput, boundingBox, Scalar(0, 0, 255), 2);

			// Desenarea contururilor individuale în verde
			for (const auto& rect : group) {
				rectangle(contourOutput, rect, Scalar(0, 255, 0), 2);
			}

			cout << "Zebra crossing detected!" << endl;
		}
	}

	return contourOutput;
}


bool isInside(Mat img, int i, int j)
{
	if ((i >= 0 && i < img.rows) && (j >= 0 && j < img.cols)) {
		return true;
	}
	else {
		return false;
	}
}

Mat inchidere(Mat src, int dim) {
	bool ok = true;
	int height = src.rows;
	int width = src.cols;
	int mij = dim / 2;

	//initializarea matricilor destinatie
	Mat dst = Mat(height, width, CV_8UC1);
	Mat dst2 = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<uchar>(i, j) = 0;
			dst2.at<uchar>(i, j) = 0;
		}
	}

	// dilatare
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 255) {
				dst.at<uchar>(i, j) = 255;
				for (int k = 0; k < dim; k++) {
					for (int t = 0; t < dim; t++) {
						if (isInside(src, i + k - mij, j + t - mij) && dst.at<uchar>(i + k - mij, j + t - mij) == 0) {
							dst.at<uchar>(i + k - mij, j + t - mij) = 255;
						}
					}
				}
			}
		}
	}

	// eroziune
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst.at<uchar>(i, j) == 255) {
				ok = false;
				for (int k = 0; k < dim; k++) {
					for (int t = 0; t < dim; t++) {
						if (!isInside(dst, i + k - mij, j + t - mij) || dst.at<uchar>(i + k - mij, j + t - mij) == 0) {
							ok = true;
						}
					}
				}
				if (ok == false) {
					dst2.at<uchar>(i, j) = 255;
				}
			}
		}
	}

	return dst2;
}

void project_demo() {
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		src = imread(fname, IMREAD_COLOR);
		imshow("Original picture", src);

		Mat gray;
		cvtColor(src, gray, COLOR_BGR2GRAY);

		imshow("Grayscale image", gray);

		Mat dst = median_filter1(gray);

		imshow("Filtered image", dst);

		Mat binarizare = Mat(dst.rows, dst.cols, CV_8UC1);
		binarizare.setTo(255);

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++) {
				if (dst.at<uchar>(i, j) < 130) {
					binarizare.at<uchar>(i, j) = 0;
				}
			}

		Mat closedImage, zebraContourImage;

		//closedImage = inchidere(binarizare, 5);
		zebraContourImage = detectZebraCrossing(binarizare, src);

		imshow("Zebra Crossing Contour", zebraContourImage);

		imshow("Binarized Image", binarizare);


		waitKey(0);
	}
}


int main() {
	int op;
	do {
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Project demo\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op) {
		case 1:
			testOpenImage();
			break;
		case 2:
			project_demo();
			break;
		}
	} while (op != 0);
	return 0;
}
