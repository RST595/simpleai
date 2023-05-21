package com.example.simpleai.onelayer.semaphore;

/**
 * for INPUT = {{1, 0, 1},
 *              {0, 1, 1},
 *              {0, 0, 1},
 *              {1, 1, 1}}
 * and OUTPUT = {1, 1, 0, 0}
 * not working well, because, there no correlation between any input and output
 *      + 0 +      ->      1
 *      0 + +      ->      1
 *      0 0 -      ->      0
 *      - - -      ->      0
 * Sum of "+" and "-" for EACH column are equals, so AI try to push weight up and down at same time
 */
public class SemaphoreMISO {
    private static final double ALPHA = 0.0001;
    private static final double[][] INPUT = {
            {1, 0, 1},
            {0, 1, 1},
            {0, 0, 1},
            {1, 1, 1},
            {0, 1, 1},
            {1, 0, 1},
    };

    private static final double[] OUTPUT = {0, 1, 0, 1, 1, 0};
    private static final double[] WEIGHTS = {0.2, 0.2, 0.2};

    public static void main(String[] args) {
        for (int k = 0; k < 10_000; k++) {
            for (int i = 0; i < INPUT.length; i++) {
                double pred = INPUT[i][0] * WEIGHTS[0] + INPUT[i][1] * WEIGHTS[1] + INPUT[i][2] * WEIGHTS[2];
                double delta = pred - OUTPUT[i];

                // double error = Math.pow(delta, 2);

                for (int j = 0; j < WEIGHTS.length; j++) {
                    double weightDelta = ALPHA * delta * INPUT[i][j];
                    WEIGHTS[j] -= weightDelta;
                }

                printResults();
            }
        }
    }

    private static void printResults() {
        System.out.println("WEIGHTS: " +
                String.format("%.2f", WEIGHTS[0]) + " " +
                String.format("%.2f", WEIGHTS[1]) + " " +
                String.format("%.2f", WEIGHTS[2])
        );

        for (int i = 0; i < OUTPUT.length; i++) {
            System.out.println("PREDICTION = " + OUTPUT[i] + "; REAL = " + String.format("%.2f", (
                            INPUT[i][0] * WEIGHTS[0] +
                            INPUT[i][1] * WEIGHTS[1] +
                            INPUT[i][2] * WEIGHTS[2]
            )));
        }

        System.out.println("--------------------------------------------------------------------");
    }
}
