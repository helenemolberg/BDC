import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.IntStream;

public class G35HW2 {


public static void main(String[] args) throws IOException {

    //Checking number of CMD parameters
    //Giving an IOException if it is not correct
    if(args.length == 0){
        throw new IllegalArgumentException("Expecting the file name on the command line");
    }

    //Setting variables
    String filename = args[0];
    ArrayList<Vector> inputPoints;
    inputPoints = readVectorsSeq(filename);

    //SPARK SETUP
    SparkConf conf = new SparkConf(true).setAppName("Homework1");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("WARN");

    //Runs the exactMPS method with the inputPoints as input
    exactMPS(inputPoints);

    // Gets number of partitions from the input
    int K = Integer.parseInt(args[1]);

    //Runs the twoApproxMPD method with the following inputs
    twoApproxMPD(inputPoints, K);

    //Runs the kCenterMPD method with the following inputs,
    kCenterMPD(inputPoints, K);

}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Auxiliary methods
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //Is running through readVectorsSeq - making a list of [double, double]
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    //receives in input a set of points S and returns the max distance between two points in S
    private static double exactMPS(ArrayList<Vector> inputPoints) {
        //Starting time
        long startTime = System.currentTimeMillis();

        //Setting variables
        double distance = 0;
        double cal;
        //Uses this to make a sublist of inputPoints
        int size = inputPoints.size();

        //Uses two for-loops to iterate through two vectors
        //to be able to compare all the elements with each other
        for(Vector vector : inputPoints){
            for(Vector vector2 : inputPoints.subList(1, size)){
                //Uses the formula for calculating distance. - sqrt((x1-x0)^2 + (y1-y0)^2)
                cal = Math.sqrt((Math.pow(vector2.apply(0)-vector.apply(0), 2) +
                        Math.pow(vector2.apply(1)-vector.apply(1), 2)));
                //Checking if the new distance is larger than the previous one
                if (cal >= distance) distance = cal;

            }
        }
        //Ending time
        long endTime = System.currentTimeMillis();

        System.out.println("EXACT ALGORITHM");
        System.out.println("Max distance = " + distance);
        System.out.println("Running time = " + (endTime - startTime) + " milliseconds" + "\n");

        //Added a return statement to be able to get the variable for the kCenterMPD
        return distance;

    }

    private static void twoApproxMPD(ArrayList<Vector> inputPoints, int K) {
        //Starting time
        long startTime = System.currentTimeMillis();

        //Setting SEED - 1219042
        long seed = 1227743; //Student-ID

        //Creating new variables
        double distance = 0;
        double cal;
        double cal_K;
        int size = inputPoints.size();

        ArrayList<Vector> K_inputPoints = new ArrayList<>();

        //Creating random generator
        Random random = new Random();
        random.setSeed(seed);

        //Checking if integer k is not larger or equal than the size of inputPoints
        //and gives information about the size so it is easier to choose another K
        if(K >= inputPoints.size()) throw new IllegalArgumentException(
                "Integer k is too large or equal to the size of inputPoints. " +
                        "It must be smaller than, " + size);

        //Creating a loop to iterate through inputPoints to collect K random points
        for(int i = 0; i < K; i++){
            //Setting a variable for random generator, so I can remove the exact element from inputPoints as well
            //Have to change the boundary since the size is changing every time
            int r = random.nextInt(inputPoints.size());
            //Uses the size of inputPoints as the boundary,
            //or the random seed can get bigger than the size of inputPoints
            K_inputPoints.add(inputPoints.get(r));

            //Removing the K elements from inputPoints
            inputPoints.remove(r);
        }

        //Calculate the maximum distance from S'
        for (Vector vectorX : K_inputPoints){
            for (Vector vectorX1 : K_inputPoints.subList(1, K_inputPoints.size())){
                //Uses the formula for calculating distance. - sqrt((x1-x0)^2 + (y1-y0)^2)
                cal_K = Math.sqrt((Math.pow(vectorX1.apply(0)-vectorX.apply(0), 2) +
                        Math.pow(vectorX1.apply(1)-vectorX.apply(1), 2)));
                if(cal_K >= distance) distance = cal_K;
            }
        }
        //Calculate the maximum distance from S
        for (Vector vectorY : inputPoints){
            for (Vector vectorY1 : inputPoints.subList(1,inputPoints.size())){
                cal = Math.sqrt((Math.pow(vectorY1.apply(0)-vectorY.apply(0), 2) +
                        Math.pow(vectorY1.apply(1)-vectorY.apply(1), 2)));
                //Comparing the max distance from S' as well
                if (cal >= distance) distance = cal;
            }
        }

        //Ending time
        long endTime = System.currentTimeMillis();

        //Have to add the elements removed from inputPoints so the next method have all the elements
        inputPoints.addAll(K_inputPoints);

        System.out.println("2-APPROXIMATION ALGORITHM");
        System.out.println("k = " + K);
        System.out.println("Max distance = " + distance);
        System.out.println("Running time = " + (endTime - startTime) + "milliseconds" + "\n");
        //If we have a high K-value we can se that this method can be faster than the previous one.
    }

    private static void kCenterMPD(ArrayList<Vector> inputPoints, int K) {
        //Starting time
        long startTime = System.currentTimeMillis();

        //Setting variables
        ArrayList<Vector> centers = new ArrayList<>(K);
        int n = inputPoints.size();
        double distance;
        ArrayList<Double> centerMinDist = new ArrayList<>(n); //track of the closest distances from center

        //Checking if integer k is not larger or equal than the size of inputPoints
        //and gives information about the size so it is easier to choose another K
        if(K >= inputPoints.size()) throw new IllegalArgumentException(
                "Integer k is too large or equal to the size of inputPoints. " +
                        "It must be smaller than, " + inputPoints.size());

        //Generating a random number p to choose a random point from input
        Random random = new Random();
        int p = random.nextInt(inputPoints.size());
        centers.add(inputPoints.get(p)); //First center chosen
        System.out.println(centers.get(0));

        //Saves the distance of each point from the first random selected point p.
        //Maybe this is not necessary?
        for(int i = 0; i < n; i++){
            centerMinDist.add(i, euclideanDist(centers.get(0), inputPoints.get(i)));
        }

        //Starting the cycle
        for(int h = 1; h < K; h++){
            //Selecting as center the point that is the farthest from an already selected center
            int maxDistIndex = IntStream.range(0, centerMinDist.size()).boxed()
                    .max(Comparator.comparing(centerMinDist::get)).orElse(-1);
            centers.add(inputPoints.get(maxDistIndex));
            //Updating min distance
            for(int j = 0; j < n; j++){
                double currentDist = euclideanDist(inputPoints.get(j), centers.get(h));
                centerMinDist.set(j, Math.min(centerMinDist.get(j), currentDist));
            }

        }
        //Getting the distance for the centers
        distance = exactMPS(centers);

        //Ending time
        long endTime = System.currentTimeMillis();

        System.out.println("k-CENTER-BASED ALGORITHM");
        System.out.println("k = " + K);
        System.out.println("Max distance = " + distance);
        System.out.println("Running time = " + ((endTime - startTime)) + "milliseconds");


    }

    //Method for calculating the Euclidean Distance
    // --------------------------------------------------------------------------------------
    public static double euclideanDist(Vector a, Vector b){
        return Math.sqrt(Vectors.sqdist(a, b));
    }

}
