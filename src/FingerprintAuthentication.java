import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class FingerprintAuthentication {

	private TextField imageNameField;
	private Button loginButton;
	private Label resultLabel;
	private Label fmrLabel;
	private Label fnmrLabel;
	private Label eerLabel;

	private static final String TRAINING_FOLDER_PATH = "C:\\Users\\EASY LIFE\\Eclipse IDE\\.metadata\\FingerprintAuthentication\\src\\training";
	private static final String TESTING_FOLDER_PATH = "C:\\Users\\EASY LIFE\\Eclipse IDE\\.metadata\\FingerprintAuthentication\\src\\testing";

	public FingerprintAuthentication() {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load OpenCV

		imageNameField = new TextField();
		imageNameField.setPromptText("Enter Your User Name");
		imageNameField.setStyle("-fx-prompt-text-fill: gray;");

		loginButton = new Button("Login");
		loginButton.setOnAction(e -> authenticate());

		resultLabel = new Label();
		resultLabel.setFont(Font.font(14));
		resultLabel.setTextFill(Color.RED);

		fmrLabel = new Label();
		fnmrLabel = new Label();
		eerLabel = new Label();
	}

	public TextField getImageNameField() {
		return imageNameField;
	}

	public Button getLoginButton() {
		return loginButton;
	}

	public Label getResultLabel() {
		return resultLabel;
	}

	public Label getFmrLabel() {
		return fmrLabel;
	}

	public Label getFnmrLabel() {
		return fnmrLabel;
	}

	public Label getEerLabel() {
		return eerLabel;
	}

	private void authenticate() {
		String imageName = imageNameField.getText().trim();
		if (imageName.isEmpty()) {
			showError("Please enter an user name.");
			return;
		}

		// load image
		String chosenImagePath = TRAINING_FOLDER_PATH + File.separator + imageName + ".tif";
		System.out.println("Chosen Image Path: " + chosenImagePath);
		Mat chosenImage = Imgcodecs.imread(chosenImagePath, Imgcodecs.IMREAD_GRAYSCALE);

		// check successfully of chosen image
		if (chosenImage.empty()) {
			showError("Error: Failed to load the chosen image");
			return;
		}

		double threshold = 0.55; // should be less than 0.55

		File trainingFolder = new File(TRAINING_FOLDER_PATH);
		File[] trainingImages = trainingFolder.listFiles();
		File testingFolder = new File(TESTING_FOLDER_PATH);

		// variables for evaluation metrics
		int genuineCount = 0;
		int impostorCount = 0;
		int genuineMatches = 0;
		int impostorMatches = 0;
		String bestMatchName = "";
		double bestDistance = Double.POSITIVE_INFINITY;

		// compare chosen image with both genuine & impostor images
		if (trainingImages != null) {
			File[] testingImages = testingFolder.listFiles();
			for (int i = 0; i < testingImages.length; i++) {
			    File testingImage = testingImages[i];
				if (testingImage.isFile()) {
					Mat existingImage = Imgcodecs.imread(testingImage.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
					double distance = calculateDistance(chosenImage, existingImage);
					System.out.println(
							"Distance with " + testingImage.getName() + ": " + String.format("%.4f", distance));

					// update best match if closer than previous best
					if (distance < bestDistance) {
						bestDistance = distance;
						bestMatchName = testingImage.getName();
					}

					// check if the image is from genuine or impostor
					if (isGenuineUser(testingImage)) {
						genuineCount++;
						if (distance <= threshold) {
							genuineMatches++;
						}
					} else {
						impostorCount++;
						if (distance > threshold) {
							impostorMatches++;
						}
					}
				}
			}
		}

		// Calculate FMR
		double fmr = 0.0;
		if (impostorCount != 0) {
			fmr = (double) impostorMatches / impostorCount;
		}

		// Calculate FNMR
		double fnmr = 0.0;
		if (genuineCount != 0) {
			fnmr = (double) (genuineCount - genuineMatches) / genuineCount;
		}

		// Calculate EER
		double eer = ((fmr + fnmr) / 2.0) * 100;

		fmrLabel.setText("False Match Rate (FMR): " + String.format("%.5f", fmr));
		fnmrLabel.setText("False Non-Match Rate (FNMR): " + String.format("%.5f", fnmr));
		eerLabel.setText("Equal Error Rate (EER): " + String.format("%.5f", eer) + " %");

		// update result labels with best match information
		if (!bestMatchName.isEmpty() && bestDistance <= threshold) {
			resultLabel.setText("Similarity found with: " + bestMatchName + " with distance: "
					+ String.format("%.4f", bestDistance));
			resultLabel.setTextFill(Color.GREEN);

		} else {
			resultLabel.setTextFill(Color.RED);

		}
	}

	// check if an image is from a genuine user or impostor
	private boolean isGenuineUser(File imageFile) {
		// check if the parent directory of the image file is the training folder
		return imageFile.getParentFile().equals(new File(TRAINING_FOLDER_PATH));
	}

	private void showError(String message) {
		resultLabel.setTextFill(Color.RED);
		resultLabel.setText(message);
	}

	private static double calculateDistance(Mat img1, Mat img2) {
		// preprocess images
		Mat processedImg1 = preprocessImage(img1);
		Mat processedImg2 = preprocessImage(img2);

		// Extract features from preprocessed images
		Mat features1 = extractFeatures(processedImg1);
		Mat features2 = extractFeatures(processedImg2);

		// Match features and calculate distance
		double distance = matchFeatures(features1, features2);

		return distance;
	}

	private static Mat preprocessImage(Mat img) {
		// resize image to a standard size
		Mat resizedImg = new Mat();
		Imgproc.resize(img, resizedImg, new org.opencv.core.Size(200, 200));

		// normalize pixel values
		Mat normalizedImg = new Mat();
		Core.normalize(resizedImg, normalizedImg, 0, 255, Core.NORM_MINMAX, CvType.CV_8U); // 0 -> 255 standard range
																							// for gray scale images

		return normalizedImg;
	}

	private static Mat extractFeatures(Mat img) {
		BRISK brisk = BRISK.create();
		MatOfKeyPoint keypoints = new MatOfKeyPoint(); // container for storing key point information
		brisk.detect(img, keypoints);

		Mat descriptors = new Mat(); // compute descriptors
		brisk.compute(img, keypoints, descriptors);

		return descriptors;
	}

	private static double matchFeatures(Mat features1, Mat features2) {
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		List<MatOfDMatch> matches = new ArrayList<>();
		matcher.knnMatch(features1, features2, matches, 2);

		// filter good matches based on distance ratio
		List<DMatch> goodMatches = new ArrayList<>();
		for (int i = 0; i < matches.size(); i++) {
		    MatOfDMatch match = matches.get(i);
			DMatch[] matchArray = match.toArray();
			if (matchArray.length >= 2) {
				if (matchArray[0].distance <= 0.9 * matchArray[1].distance) {
					goodMatches.add(matchArray[0]);
				}
			}
		}

		// Calculate distance based on the number of good matches
		double distance = 1.0 - ((double) goodMatches.size() / Math.max(features1.rows(), features2.rows()));

		return distance;
	}
}
