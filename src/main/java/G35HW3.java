import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.*;
import java.util.stream.IntStream;

public class G35HW3 {

    public static void main(String[] args) {
        //Checking number of CMD parameters
        //Gonna get filename, k and L
        if(args.length == 0){
            throw new IllegalArgumentException("NEED: filename k L");
        }

        //Gets the filename
        String filename = args[0];
        // Gets parameter for diversity maximization
        int K = Integer.parseInt(args[1]);
        // Gets number of partitions from the input
        int L = Integer.parseInt(args[2]);

        //SPARK SETUP
        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //PARSING INPUT FILE
        long start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(filename)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long end = System.currentTimeMillis();

        long N = inputPoints.count();

        System.out.println("Number of points = " + N);
        System.out.println("K = " + K);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (end - start) + " ms");

        //Runs runMapReduce
        ArrayList<Vector> pointsSet = runMapReduce(inputPoints, K, L);

        //Calculating the average distance from the solution from runMapReduce
        double averageDist = measure(pointsSet);
        System.out.println("\nAverage distance = " + averageDist);


    }

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L){

        //------------------------- ROUND 1 ---------------------------------------------------------------
        //subdivide pointsRDD into L partitions and extract k points from each partition using
        //the Farthest-First Traversal algorithm

        //Starting time for Round 1
        long start1 = System.currentTimeMillis();

        JavaRDD<Vector> pointS = pointsRDD.mapPartitions(x ->{
           ArrayList<Vector> points = new ArrayList<>();
           while(x.hasNext()){
               points.add(x.next());
           }
           //Recycle the Farthest-First Traversal algorithm from HW2
           ArrayList<Vector> centers = kCenterMPD(points, k);

           return centers.iterator();
        });

        //Ending time from Round 1
        long stop1 = System.currentTimeMillis();
        System.out.println("\nRuntime of Round 1 = " + (stop1-start1) + " ms");

        //------------------------- ROUND 2 ---------------------------------------------------------------
        /*collects the L*k points extracted in Round 1 from the partitions into a set called coreset and returns,
        as output, the k points computed by runSequential(coreset,k). Note that coreset is not an RDD
        but an ArrayList<Vector> (in Java) or a list of tuple (in Python)
         */

        //Starting time for Round 2
        long start2 = System.currentTimeMillis();

        ArrayList<Vector> coreset = new ArrayList<>(k*L);
        coreset.addAll(pointS.collect());
        System.out.println(coreset);

        ArrayList<Vector> pointsSet = runSequential(coreset, k);
        System.out.println(pointsSet);

        long stop2 = System.currentTimeMillis();
        System.out.println("Runtime of Round 2 = " + (stop2-start2) + " ms");

        return pointsSet;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

    //Determines the average distance among the solution points
    //---------------------------------------------------------------------------------------
    public static double measure(ArrayList<Vector> pointsSet){
        double averageDist = 0;

        return averageDist;
    }

    //From HW2 - The Farthest-First Traversal algorithm
    //---------------------------------------------------------------------------------------
    private static ArrayList<Vector> kCenterMPD(ArrayList<Vector> inputPoints, int K) {

        //Setting variables
        ArrayList<Vector> centers = new ArrayList<>(K);
        int n = inputPoints.size();
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

        return centers;
    }

    //Method for calculating the Euclidean Distance
    // --------------------------------------------------------------------------------------
    public static double euclideanDist(Vector a, Vector b){
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    // From HW2 - used as function in map to create a Vector
    // --------------------------------------------------------------------------------------
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

}
