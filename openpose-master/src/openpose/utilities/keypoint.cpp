#include <stdio.h> 
#include <math.h> 
#include <limits> // std::numeric_limits
#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/gui/parameter.hpp>
bool is_neck = false;
std::string shouder_vec = " ";
std::string distance_vec = " ";
std::string knee_distance_vec = " ";
std::string back_vec = " ";
std::string warn_str = "";

namespace op
{
	const std::string errorMessage = "The Array<float> is not a RGB image. This function is only for array of"
		" dimension: [sizeA x sizeB x 3].";

	float getDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB)
	{
		try
		{
			const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
			const auto pixelX = keypointPtr[elementA * 3] - keypointPtr[elementB * 3];
			const auto pixelY = keypointPtr[elementA * 3 + 1] - keypointPtr[elementB * 3 + 1];
			return std::sqrt(pixelX*pixelX + pixelY*pixelY);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return -1.f;
		}
	}

	void averageKeypoints(Array<float>& keypointsA, const Array<float>& keypointsB, const int personA)
	{
		try
		{
			// Security checks
			if (keypointsA.getNumberDimensions() != keypointsB.getNumberDimensions())
				error("keypointsA.getNumberDimensions() != keypointsB.getNumberDimensions().",
					__LINE__, __FUNCTION__, __FILE__);
			for (auto dimension = 1u; dimension < keypointsA.getNumberDimensions(); dimension++)
				if (keypointsA.getSize(dimension) != keypointsB.getSize(dimension))
					error("keypointsA.getSize() != keypointsB.getSize().", __LINE__, __FUNCTION__, __FILE__);
			// For each body part
			const auto numberParts = keypointsA.getSize(1);
			for (auto part = 0; part < numberParts; part++)
			{
				const auto finalIndexA = keypointsA.getSize(2)*(personA*numberParts + part);
				const auto finalIndexB = keypointsA.getSize(2)*part;
				if (keypointsB[finalIndexB + 2] - keypointsA[finalIndexA + 2] > 0.05f)
				{
					keypointsA[finalIndexA] = keypointsB[finalIndexB];
					keypointsA[finalIndexA + 1] = keypointsB[finalIndexB + 1];
					keypointsA[finalIndexA + 2] = keypointsB[finalIndexB + 2];
				}
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void scaleKeypoints(Array<float>& keypoints, const float scale)
	{
		try
		{
			scaleKeypoints(keypoints, scale, scale);
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY)
	{
		try
		{
			if (scaleX != 1. && scaleY != 1.)
			{
				// Error check
				if (!keypoints.empty() && keypoints.getSize(2) != 3)
					error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
				// Get #people and #parts
				const auto numberPeople = keypoints.getSize(0);
				const auto numberParts = keypoints.getSize(1);
				// For each person
				for (auto person = 0; person < numberPeople; person++)
				{
					// For each body part
					for (auto part = 0; part < numberParts; part++)
					{
						const auto finalIndex = 3 * (person*numberParts + part);
						keypoints[finalIndex] *= scaleX;
						keypoints[finalIndex + 1] *= scaleY;



					}
				}
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX,
		const float offsetY)
	{
		try
		{
			if (scaleX != 1. && scaleY != 1.)
			{
				// Error check
				if (!keypoints.empty() && keypoints.getSize(2) != 3)
					error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
				// Get #people and #parts
				const auto numberPeople = keypoints.getSize(0);
				const auto numberParts = keypoints.getSize(1);
				// For each person
				for (auto person = 0; person < numberPeople; person++)
				{
					// For each body part
					for (auto part = 0; part < numberParts; part++)
					{
						const auto finalIndex = keypoints.getSize(2)*(person*numberParts + part);
						keypoints[finalIndex] = keypoints[finalIndex] * scaleX + offsetX;
						keypoints[finalIndex + 1] = keypoints[finalIndex + 1] * scaleY + offsetY;
					}
				}
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
	///
	double countAngle(int x1, int y1, int x2, int y2, int x3, int y3) {
		double PI = 3.14159265;
		double cos_param, result;

		int BA_x, BA_y, BC_x, BC_y;
		BA_x = x1 - x2;
		BA_y = y1 - y2;
		BC_x = x3 - x2;
		BC_y = y3 - y2;
		cos_param = (BA_x*BC_x + BA_y*BC_y) / (sqrt(pow(BA_x, 2) + pow(BA_y, 2))*sqrt(pow(BC_x, 2) + pow(BC_y, 2)));
		result = acos(cos_param) * 180.0 / PI;
		return result;
		//printf("The arc cosine of %f is %f degrees.\n", param, result);
	}
	///
	double countDistance(int x1, int y1, int x2, int y2) {
		double distance;
		int BA_x, BA_y;
		BA_x = x1 - x2;
		BA_y = y1 - y2;
		distance = sqrt(BA_x*BA_x + BA_y*BA_y);
		return distance;
	}


	void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints,
		const std::vector<unsigned int>& pairs, const std::vector<float> colors,
		const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
		const float threshold)
	{
		try
		{
			if (!frameArray.empty())
			{
				// Array<float> --> cv::Mat
				auto frame = frameArray.getCvMat();


				// Security check
				if (frame.dims != 3 || frame.size[0] != 3)
					error(errorMessage, __LINE__, __FUNCTION__, __FILE__);

				// Get frame channels
				const auto width =frame.size[2];
				const auto height =frame.size[1];
				const auto area = width * height;
				cv::Mat frameB(height, width, CV_32FC1, &frame.data[0]);
				cv::Mat frameG(height, width, CV_32FC1, &frame.data[area * sizeof(float) / sizeof(uchar)]);
				cv::Mat frameR(height, width, CV_32FC1, &frame.data[2 * area * sizeof(float) / sizeof(uchar)]);

				// Parameters
				const auto lineType = 8;
				const auto shift = 0;
				const auto numberColors = colors.size();
				const auto thresholdRectangle = 0.1f;
				const auto numberKeypoints = keypoints.getSize(1);
				bool is_one_time = false;

				int theTrainPerson, theTrainPerson1 = 11, theTrainPerson2 = 11;
				int temp_distance;
				int top_min_distance = 1920;
				int down_min_distance = 1920;

				if (is_start == true) {


					for (auto person = 0; person < keypoints.getSize(0); person++)
					{
						const auto person_temp_Rectangle = getKeypointsRectangle(keypoints, person, thresholdRectangle);
						temp_distance = abs((person_temp_Rectangle.x + person_temp_Rectangle.width / 4) - (width / 4));
						if (person_temp_Rectangle.area() > 0) {

							if ((person_temp_Rectangle.y  <  height / 2) && temp_distance < top_min_distance) {
								top_min_distance = temp_distance;
								theTrainPerson1 = 0;
							}
							else if (((person_temp_Rectangle.y) >  height / 2) && temp_distance < down_min_distance) {
								down_min_distance = temp_distance;
								theTrainPerson2 = person;
							}
						}
					}


					if (keypoints.getSize(0) > 0) {

						is_one_time = false;

						for (auto num = 0; num < 2; num++)
						{

							if (is_one_time == true)break;

							if (theTrainPerson1 > 4 && theTrainPerson2 > 4) {
								break;
							}
							else if (theTrainPerson1 < 10 && theTrainPerson2 > 10) {
								is_one_time = true;
								theTrainPerson = theTrainPerson1;
							}
							else if (theTrainPerson1 > 10 && theTrainPerson2 < 10) {
								is_one_time = true;
								theTrainPerson = theTrainPerson2;
							}
							else if (num == 0) {
								theTrainPerson = theTrainPerson1;
							}
							else {
								theTrainPerson = theTrainPerson2;
							}



							//if(num==0)theTrainPerson = theTrainPerson1;
							//else theTrainPerson = theTrainPerson2;


							printf("theTrainPerson: %d\n", theTrainPerson);

							// Keypoints
							// for (auto person = 0 ; person < keypoints.getSize(0) ; person++){
							const auto personRectangle = getKeypointsRectangle(keypoints, theTrainPerson, thresholdRectangle);


							if (personRectangle.area() > 0)
							{
								const auto ratioAreas = fastMin(1.f, fastMax(personRectangle.width / (float)width,
									personRectangle.height / (float)height));
								// Size-dependent variables
								const auto thicknessRatio = fastMax(intRound(std::sqrt(area)
									* thicknessCircleRatio * ratioAreas), 2);
								// Negative thickness in cv::circle means that a filled circle is to be drawn.
								const auto thicknessCircle = (ratioAreas > 0.05 ? thicknessRatio : -1);
								const auto thicknessLine = intRound(thicknessRatio * thicknessLineRatioWRTCircle);
								const auto radius = thicknessRatio / 2;

								int pairNum = 0;


								/*pairNum*/
								/* 1: 1,2 Rshouder
								2: 1.5
								3: 2,3
								4: 3.4
								5: 5.6
								6: 6.7
								7: 1.8
								8: 8.9
								9: 9.10
								10: 1.11
								11: 11.12
								12: 12.13
								13: 1.0
								14: 0.14
								15: 14.16
								16: 0.15
								17: 15.11
								18: 2.16
								19: 5.17
								*/

								if (theTrainPerson == theTrainPerson2) {//down
																		// Draw lines
									for (auto pair = 0u + 12; pair < 0u + 18; pair += 2)
									{
										const auto index1 = (theTrainPerson * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
										const auto index2 = (theTrainPerson * numberKeypoints + pairs[pair + 1]) * keypoints.getSize(2);
										if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
										{
											const auto colorIndex = pairs[pair + 1] * 3; // Before: colorIndex = pair/2*3;
											const cv::Scalar color{ colors[colorIndex % numberColors],
												colors[(colorIndex + 1) % numberColors],
												colors[(colorIndex + 2) % numberColors] };
											const cv::Point keypoint1{ intRound(keypoints[index1]), intRound(keypoints[index1 + 1]) };
											const cv::Point keypoint2{ intRound(keypoints[index2]), intRound(keypoints[index2 + 1]) };
											if (keypoint1.y > (height / 2) && keypoint2.y > (height / 2) && keypoint1.x >0 && keypoint2.x >0) {
												//cv::line(InputOutputArray img,Point pt1, Point pt2, const Scalar & color, int thickness = 1,int lineType = LINE_8, int shift = 0)
												cv::line(frameR, keypoint1, keypoint2, color[0], thicknessLine, lineType, shift);
												cv::line(frameG, keypoint1, keypoint2, color[1], thicknessLine, lineType, shift);
												cv::line(frameB, keypoint1, keypoint2, color[2], thicknessLine, lineType, shift);
											}
											pairNum++;
										}
									}
								}
								else if (theTrainPerson == theTrainPerson1) {//up

									for (auto pair = 0u; pair < pairs.size(); pair += 2)
									{
										const auto index1 = (theTrainPerson * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
										const auto index2 = (theTrainPerson * numberKeypoints + pairs[pair + 1]) * keypoints.getSize(2);
										if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
										{
											const auto colorIndex = pairs[pair + 1] * 3; // Before: colorIndex = pair/2*3;
											const cv::Scalar color{ colors[colorIndex % numberColors],
												colors[(colorIndex + 1) % numberColors],
												colors[(colorIndex + 2) % numberColors] };
											const cv::Point keypoint1{ intRound(keypoints[index1]), intRound(keypoints[index1 + 1]) };
											const cv::Point keypoint2{ intRound(keypoints[index2]), intRound(keypoints[index2 + 1]) };

											if (keypoint1.y < height / 2 && keypoint2.y < height / 2 && keypoint1.x > 0 && keypoint2.x > 0) {
												//cv::line(InputOutputArray img,Point pt1, Point pt2, const Scalar & color, int thickness = 1,int lineType = LINE_8, int shift = 0)
												cv::line(frameR, keypoint1, keypoint2, color[0], thicknessLine, lineType, shift);
												cv::line(frameG, keypoint1, keypoint2, color[1], thicknessLine, lineType, shift);
												cv::line(frameB, keypoint1, keypoint2, color[2], thicknessLine, lineType, shift);
											}

											pairNum++;
										}
									}
								}





								int posePoint_X[18];
								int posePoint_Y[18];
								cv::Scalar colorRightEar(0, 0, 255);
								cv::Point neck;


								// Draw circles
								for (auto part = 0; part < numberKeypoints; part++)
								{
									const auto faceIndex = (theTrainPerson * numberKeypoints + part) * keypoints.getSize(2);
									if (keypoints[faceIndex + 2] > threshold)
									{
										const auto colorIndex = part * 3;
										const cv::Scalar color{ colors[colorIndex % numberColors],
											colors[(colorIndex + 1) % numberColors],
											colors[(colorIndex + 2) % numberColors] };
										const cv::Point center{ intRound(keypoints[faceIndex]),
											intRound(keypoints[faceIndex + 1]) };

										posePoint_X[part] = center.x;
										posePoint_Y[part] = center.y;

										if (num == 1 && center.y > (height / 2) && center.x > 0) {


											if (part == 16 || part == 1 || part == 8 || part == 9 || part == 10) {
												cv::circle(frameR, center, radius, color[0], thicknessCircle, lineType, shift);
												cv::circle(frameG, center, radius, color[1], thicknessCircle, lineType, shift);
												cv::circle(frameB, center, radius, color[2], thicknessCircle, lineType, shift);
											}

											if (part == 1 && center.y > (height / 2) && center.x > 0) {
												is_neck = true;
												neck = center;
											}
											else if (part == 1) {
												is_neck = false;
											}

											//posePoint_X[part] = center.x;
											//posePoint_Y[part] = center.y;

											if (part == 16 && center.y > height / 2 && center.x > 0 && is_neck == true) {
												colorRightEar = color;
												//cv::putText(frameR, "Nose", center, 0, 1, color, 3);
												//cv::putText(frameG, "Nose", center, 0, 1, color, 3);
												//cv::putText(frameB, "Nose", center, 0, 1, color, 3);
												//printf("0000000000000\n");
												cv::line(frameR, center, neck, colorRightEar[0], thicknessLine, lineType, shift);
												cv::line(frameG, center, neck, colorRightEar[1], thicknessLine, lineType, shift);
												cv::line(frameB, center, neck, colorRightEar[2], thicknessLine, lineType, shift);
											}
										}
										else if (num == 0 && center.y < height / 2 && center.x >0 && center.y >0) {
											cv::circle(frameR, center, radius, color[0], thicknessCircle, lineType, shift);
											cv::circle(frameG, center, radius, color[1], thicknessCircle, lineType, shift);
											cv::circle(frameB, center, radius, color[2], thicknessCircle, lineType, shift);
											posePoint_X[part] = center.x;
											posePoint_Y[part] = center.y;
										}


									}
								}






								printf("-----------------------------------------\n");
								for (int part = 0; part < numberKeypoints; part++) {
									printf("posePoint_X[%d] = %d , posePoint_Y[%d] = %d\n", part, posePoint_X[part], part, posePoint_Y[part]);

								}


								cv::Point center;
								cv::Scalar color(255, 0, 0);
								char output[50];
								shouder_vec = " ";
								// hand_vec = " ";
								distance_vec = " ";
								knee_distance_vec = " ";
								back_vec = " ";
								warn_str = "";

								//Shouder_horizontal_line
								//2 RShouder 5 LShouder 正

								if (theTrainPerson == theTrainPerson1) {

									if (posePoint_X[2] > 0 && posePoint_Y[2] > 0 && posePoint_X[5] > 0 && posePoint_Y[5]) {
										double num = posePoint_Y[5] - posePoint_Y[2];
										//double num = countAngle(posePoint_X[5], posePoint_Y[5], posePoint_X[2], posePoint_Y[2], posePoint_X[2] + 30, posePoint_Y[2]);
										//double num1 = countAngle(posePoint_X[2], posePoint_Y[2], posePoint_X[5], posePoint_Y[5], posePoint_X[5] + 30, posePoint_Y[5]);
										
										//shouder_vec += std::to_string(num_X) ;
										if (num > hand_height_difference) {
											shouder_vec += "左肩掉下來囉 !";
										}
										else if (num < (-1 * hand_height_difference)) {
											shouder_vec += "右肩掉下來囉 !";
										}
									}

									/*
									if (posePoint_X[4] > 0 && posePoint_Y[7] > 0 && posePoint_X[4] > 0 && posePoint_Y[7]) {
									//int num_X = posePoint_X[5] - posePoint_X[2];
									int num_Y = posePoint_Y[7] - posePoint_Y[4];
									if (num_Y <= -15) {
									hand_vec += "your RIGHT hand is too low !";
									}
									else if (num_Y >= 15) {
									hand_vec += "your LEFT hand is too low !";
									}
									center.x = posePoint_X[7] + 50;
									center.y = posePoint_Y[7] + 50;
									//cv::Scalar color_LINE(255, 225, 225);
									cv::putText(frameR, hand_vec, center, 0, 0.8, color, 2);
									cv::putText(frameG, hand_vec, center, 0, 0.8, color, 2);
									cv::putText(frameB, hand_vec, center, 0, 0.8, color, 2);
									}

									*/

									//countDistance
									if (posePoint_X[2] > 0 && posePoint_Y[2] > 0 && posePoint_X[5] > 0 && posePoint_Y[5] > 0 && posePoint_X[13] > 0 && posePoint_Y[13] > 0 && posePoint_X[10] > 0 && posePoint_Y[10] > 0) {
										double shouder_distance = countDistance(posePoint_X[2], posePoint_Y[2], posePoint_X[5], posePoint_Y[5]);
										double ankle_distance = countDistance(posePoint_X[13], posePoint_Y[13], posePoint_X[10], posePoint_Y[10]);
										if (shouder_distance > ankle_distance) {
											distance_vec += "雙腳請打開一點~";
										}
									}



									//countHip
									if (posePoint_X[11] > 0 && posePoint_Y[11] > 0 && posePoint_X[8] > 0 && posePoint_Y[8] > 0 && posePoint_X[12] > 0 && posePoint_Y[12] > 0 && posePoint_X[9] > 0 && posePoint_Y[9] > 0) {
										double hip_distance = countDistance(posePoint_X[11], posePoint_Y[11], posePoint_X[8], posePoint_Y[8]);
										double knee_distance = countDistance(posePoint_X[12], posePoint_Y[12], posePoint_X[9], posePoint_Y[9]);
										if (knee_distance < hip_distance) {
											knee_distance_vec += "膝蓋請打開一點~";
										}
									}
																	
									//RShouder
									if (posePoint_X[1] > 0 && posePoint_Y[1] > 0 && posePoint_X[5] > 0 && posePoint_Y[5] > 0 && posePoint_X[6] > 0 && posePoint_Y[6] > 0) {
										double num = countAngle(posePoint_X[1], posePoint_Y[1], posePoint_X[5], posePoint_Y[5], posePoint_X[6], posePoint_Y[6]);
										LShouder_Angle = num;
									}

									//RElbow
									if (posePoint_X[5] > 0 && posePoint_Y[5] > 0 && posePoint_X[6] > 0 && posePoint_Y[6] > 0 && posePoint_X[7] > 0 && posePoint_Y[7] > 0) {
										double num = countAngle(posePoint_X[5], posePoint_Y[5], posePoint_X[6], posePoint_Y[6], posePoint_X[7], posePoint_Y[7]);
										LElbow_Angle = num;
									}

									//LShouder
									if (posePoint_X[1] > 0 && posePoint_Y[1] > 0 && posePoint_X[2] > 0 && posePoint_Y[2] > 0 && posePoint_X[3] > 0 && posePoint_Y[3] > 0) {
										double num = countAngle(posePoint_X[1], posePoint_Y[1], posePoint_X[2], posePoint_Y[2], posePoint_X[3], posePoint_Y[3]);
										RShouder_Angle = num;
										/*
										center.x = posePoint_X[2];
										center.y = posePoint_Y[2];
										cv::putText(frameR, output, center, 0, 0.5, color, 2);
										cv::putText(frameG, output, center, 0, 0.5, color, 2);
										cv::putText(frameB, output, center, 0, 0.5, color, 2);
										printf("LShouder_Angle = %s\n", output);
										*/
									}

									//LElbow
									if (posePoint_X[2] > 0 && posePoint_Y[2] > 0 && posePoint_X[3] > 0 && posePoint_Y[3] > 0 && posePoint_X[4] > 0 && posePoint_Y[4] > 0) {
										double num = countAngle(posePoint_X[2], posePoint_Y[2], posePoint_X[3], posePoint_Y[3], posePoint_X[4], posePoint_Y[4]);
										RElbow_Angle = num;
										/*
										center.x = posePoint_X[3];
										center.y = posePoint_Y[3];
										cv::putText(frameR, output, center, 0, 0.5, color, 2);
										cv::putText(frameG, output, center, 0, 0.5, color, 2);
										cv::putText(frameB, output, center, 0, 0.5, color, 2);
										printf("LElbow_Angle = %s\n", output);
										*/
									}

									//RKnee
									if (posePoint_X[11] > 0 && posePoint_Y[11] > 0 && posePoint_X[12] > 0 && posePoint_Y[12] > 0 && posePoint_X[13] > 0 && posePoint_Y[13] > 0) {
										double num = countAngle(posePoint_X[11], posePoint_Y[11], posePoint_X[12], posePoint_Y[12], posePoint_X[13], posePoint_Y[13]);
										LKnee_Angle = num;
										/*
										center.x = posePoint_X[12];
										center.y = posePoint_Y[12];
										cv::putText(frameR, output, center, 0, 0.5, color, 2);
										cv::putText(frameG, output, center, 0, 0.5, color, 2);
										cv::putText(frameB, output, center, 0, 0.5, color, 2);
										printf("RKnee_Angle = %s\n", output);*/
									}

									//LKnee
									if (posePoint_X[8] > 0 && posePoint_Y[8] > 0 && posePoint_X[9] > 0 && posePoint_Y[9] > 0 && posePoint_X[10] > 0 && posePoint_Y[10] > 0) {
										double num = countAngle(posePoint_X[8], posePoint_Y[8], posePoint_X[9], posePoint_Y[9], posePoint_X[10], posePoint_Y[10]);
										RKnee_Angle = num;
										/*
										center.x = posePoint_X[9];
										center.y = posePoint_Y[9];
										cv::putText(frameR, output, center, 0, 0.5, color, 2);
										cv::putText(frameG, output, center, 0, 0.5, color, 2);
										cv::putText(frameB, output, center, 0, 0.5, color, 2);
										printf("LKnee_Angle = %s\n", output);*/
									}
								}



								if (theTrainPerson == theTrainPerson2) {

									//RHip
									/*if (posePoint_X[1] > 0 && posePoint_Y[1] > 0 && posePoint_X[11] > 0 && posePoint_Y[11] > 0 && posePoint_X[12] > 0 && posePoint_Y[12] > 0) {
									double num = countAngle(posePoint_X[1], posePoint_Y[1], posePoint_X[11], posePoint_Y[11], posePoint_X[12], posePoint_Y[12]);
									snprintf(output, 50, "%.2f", num);
									center.x = posePoint_X[11];
									center.y = posePoint_Y[11];
									cv::putText(frameR, output, center, 0, 0.5, color, 2);
									cv::putText(frameG, output, center, 0, 0.5, color, 2);
									cv::putText(frameB, output, center, 0, 0.5, color, 2);
									printf("RHip_Angle = %s\n", output);
									}
									//LHip
									if (posePoint_X[1] > 0 && posePoint_Y[1] > 0 && posePoint_X[8] > 0 && posePoint_Y[8] > 0 && posePoint_X[9] > 0 && posePoint_Y[9] > 0) {
									double num = countAngle(posePoint_X[1], posePoint_Y[1], posePoint_X[8], posePoint_Y[8], posePoint_X[9], posePoint_Y[9]);
									snprintf(output, 50, "%.2f", num);
									center.x = posePoint_X[8];
									center.y = posePoint_Y[8];
									cv::putText(frameR, output, center, 0, 0.5, color, 2);
									cv::putText(frameG, output, center, 0, 0.5, color, 2);
									cv::putText(frameB, output, center, 0, 0.5, color, 2);
									printf("LHip_Angle = %s\n", output);
									}
									*/

									/*//LHip
									if (posePoint_X[1] > 0 && posePoint_Y[1] > 0 && posePoint_X[16] > 0 && posePoint_Y[16] > 0 && posePoint_X[8] > 0 && posePoint_Y[8] > 0) {
									double num = countAngle(posePoint_X[16], posePoint_Y[16], posePoint_X[1], posePoint_Y[1], posePoint_X[8], posePoint_Y[8]);
									//back_vec += std::to_string(num);
									if (num <= 170) {
									back_vec += "straighten the neck";
									center.x = 830;
									center.y = 990;
									cv::putText(frameR, back_vec, center, 0, 0.8, cv::Scalar(225, 225, 225), 2);
									cv::putText(frameG, back_vec, center, 0, 0.8, cv::Scalar(225, 225, 225), 2);
									cv::putText(frameB, back_vec, center, 0, 0.8, cv::Scalar(225, 225, 225), 2);
									}
									}*/



									//countDistance
									if (posePoint_X[9] > 0 && posePoint_Y[9] > 0 && posePoint_X[10] > 0 && posePoint_Y[10] > 0 && posePoint_X[8] > 0 && posePoint_Y[8] > 0) {
										double shouder_distance = sqrt(pow(posePoint_X[9] - posePoint_X[10], 2));
										double ankle_distance = countDistance(posePoint_X[8], posePoint_Y[8], posePoint_X[9], posePoint_Y[9]);
										//distance_vec += std::to_string(shouder_distance/ankle_distance);
										if ((shouder_distance / ankle_distance) >= 0.6) {
											back_vec += "屁股請往後坐~";
										}
									}

								}


								if (shouder_vec != " ") {
									warn_str += shouder_vec + '\n';
								}
								if (knee_distance_vec != " ") {
									warn_str += knee_distance_vec + '\n';
								}
								if (distance_vec != " ") {
									warn_str += distance_vec + '\n';
								}
								if (back_vec != " ") {
									warn_str += back_vec + '\n';
								}

							}
						}//num
					} //getSized(0)

				}//setting is_rendering
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	Rectangle<float> getKeypointsRectangle(const Array<float>& keypoints, const int person, const float threshold)
	{
		try
		{
			const auto numberKeypoints = keypoints.getSize(1);
			// Security checks
			if (numberKeypoints < 1)
				error("Number body parts must be > 0", __LINE__, __FUNCTION__, __FILE__);
			// Define keypointPtr
			const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
			float minX = std::numeric_limits<float>::max();
			float maxX = 0.f;
			float minY = minX;
			float maxY = maxX;
			for (auto part = 0; part < numberKeypoints; part++)
			{
				const auto score = keypointPtr[3 * part + 2];
				if (score > threshold)
				{
					const auto x = keypointPtr[3 * part];
					const auto y = keypointPtr[3 * part + 1];
					// Set X
					if (maxX < x)
						maxX = x;
					if (minX > x)
						minX = x;
					// Set Y
					if (maxY < y)
						maxY = y;
					if (minY > y)
						minY = y;
				}
			}
			if (maxX >= minX && maxY >= minY)
				return Rectangle<float>{minX, minY, maxX - minX, maxY - minY};
			else
				return Rectangle<float>{};
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return Rectangle<float>{};
		}
	}

	float getAverageScore(const Array<float>& keypoints, const int person)
	{
		try
		{
			// Security checks
			if (person >= keypoints.getSize(0))
				error("Person index out of bounds.", __LINE__, __FUNCTION__, __FILE__);
			// Get average score
			auto score = 0.f;
			const auto numberKeypoints = keypoints.getSize(1);
			const auto area = numberKeypoints * keypoints.getSize(2);
			const auto personOffset = person * area;
			for (auto part = 0; part < numberKeypoints; part++)
				score += keypoints[personOffset + part*keypoints.getSize(2) + 2];
			return score / numberKeypoints;
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return 0.f;
		}
	}

	float getKeypointsArea(const Array<float>& keypoints, const int person, const float threshold)
	{
		try
		{
			return getKeypointsRectangle(keypoints, person, threshold).area();
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return 0.f;
		}
	}

	int getBiggestPerson(const Array<float>& keypoints, const float threshold)
	{
		try
		{
			if (!keypoints.empty())
			{
				const auto numberPeople = keypoints.getSize(0);
				auto biggestPoseIndex = -1;
				auto biggestArea = -1.f;
				for (auto person = 0; person < numberPeople; person++)
				{
					const auto newPersonArea = getKeypointsArea(keypoints, person, threshold);
					if (newPersonArea > biggestArea)
					{
						biggestArea = newPersonArea;
						biggestPoseIndex = person;
					}
				}
				return biggestPoseIndex;
			}
			else
				return -1;
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
			return -1;
		}
	}
}
