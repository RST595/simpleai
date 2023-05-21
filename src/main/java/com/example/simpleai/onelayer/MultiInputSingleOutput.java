package com.example.simpleai.onelayer;

public class MultiInputSingleOutput {
    private static final int INPUT_SIZE = 4;
    private static final double ALPHA = 0.0001;
    private static final double[][] INPUT = {{8.5, 9.5, 9.0, 9.9}, {0.65, 0.8, 0.8, 0.9}, {1.2, 1.3, 0.5, 1.0}};
    private static final double[] INPUT_1 = {8.5, 9.5, 9.0, 9.9};
    private static final double[] INPUT_2 = {0.65, 0.8, 0.8, 0.9};
    private static final double[] INPUT_3 = {1.2, 1.3, 0.5, 1.0};
    private static final double[] OUTPUT = {1, 1, 0, 1};
    private static final double[] WEIGHTS = {0.2, 0.2, 0.2};
    public static final String PREDICTION_1_REAL = "PREDICTION = 1; REAL = ";

    public static void main(String[] args) {
        for (int k = 0; k < 100_000; k++) {
            for (int i = 0; i < INPUT_SIZE; i++) {
                double pred = INPUT_1[i] * WEIGHTS[0] + INPUT_2[i] * WEIGHTS[1] + INPUT_3[i] * WEIGHTS[2];
                double delta = pred - OUTPUT[i];

                double error = Math.pow(delta, 2);

                for (int j = 0; j < 3; j++) {
                    double weightDelta = ALPHA * delta * INPUT[j][i];
                    WEIGHTS[j] -= weightDelta;
                }

                printResults();
            }
        }
    }

    private static void printResults() {
        System.out.println("WEIGHTS: " + WEIGHTS[0] + " " + WEIGHTS[1] + " " + WEIGHTS[2]);
        System.out.println(PREDICTION_1_REAL + (INPUT_1[0] * WEIGHTS[0] + INPUT_2[0] * WEIGHTS[1] + INPUT_3[0] * WEIGHTS[2]));
        System.out.println(PREDICTION_1_REAL + (INPUT_1[1] * WEIGHTS[0] + INPUT_2[1] * WEIGHTS[1] + INPUT_3[1] * WEIGHTS[2]));
        System.out.println("PREDICTION = 0; REAL = " + (INPUT_1[2] * WEIGHTS[0] + INPUT_2[2] * WEIGHTS[1] + INPUT_3[2] * WEIGHTS[2]));
        System.out.println(PREDICTION_1_REAL + (INPUT_1[3] * WEIGHTS[0] + INPUT_2[3] * WEIGHTS[1] + INPUT_3[3] * WEIGHTS[2]));
        System.out.println("--------------------------------------------------------------------");
    }
}
