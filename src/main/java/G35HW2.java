import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class G35HW2 {


public static void main(String[] args) throws IOException {

    //Checking number of CMD parameters
    //Giving an IOException if it is not correct
    if(args.length == 0){
        throw new IllegalArgumentException("Expecting the file name on the command line");
    }


    String filename = args[0];
    ArrayList<Vector> inputPoints = new ArrayList<>();
    inputPoints = readVectorsSeq(filename);

    //SPARK SETUP
    SparkConf conf = new SparkConf(true).setAppName("Homework1");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("WARN");

    System.out.println(inputPoints);

    G35HW2.exactMPS(inputPoints);


}


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Auxiliary methods
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
            //System.out.println(data[i]);
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
        //System.out.println(result);
        return result;
    }

    //receives in input a set of points S and returns the max distance between two points in S
    private static void exactMPS(ArrayList<Vector> inputPoints) {

        System.out.println("EXACT ALGORITHM");
        System.out.println("Max distance = ");
        System.out.println("Running time = ");
    }
}
