import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.VBox;
import javafx.geometry.Insets;
import javafx.stage.Stage;

public class Main extends Application {

	public static void main(String[] args) {
		launch(args);
	}

	@Override
	public void start(Stage primaryStage) {
		primaryStage.setTitle("Fingerprint Authentication");

		FingerprintAuthentication fingerprintAuthentication = new FingerprintAuthentication();
		VBox layout = new VBox(10);
		layout.setPadding(new Insets(10));
		layout.getChildren().addAll(fingerprintAuthentication.getImageNameField(),
				fingerprintAuthentication.getLoginButton(), fingerprintAuthentication.getResultLabel(),
				fingerprintAuthentication.getFmrLabel(), fingerprintAuthentication.getFnmrLabel(),
				fingerprintAuthentication.getEerLabel());

		Scene scene = new Scene(layout, 480, 250);
		primaryStage.setScene(scene);
		primaryStage.show();
	}
}
