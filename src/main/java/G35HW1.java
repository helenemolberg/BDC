import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.regex.Pattern;
import java.util.*;

public class G35HW1 {

    public static void main(String[] args) throws IOException{

        //Checking number of CMD line parameters
        //Parameters are: number_partitions, <path to file>

        if(args.length == 0){
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        //SPARK SETUP
        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Gets number of partitions from the input
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> docs = sc.textFile(args[1]).cache();

        //Setting variable
        long numdocs;

        //Setting alphabet variable
        String alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        numdocs = docs.count();
        System.out.println("Number of documents = " + numdocs);

        //Partition the the RDD in K different partitions
        docs.repartition(K);

        JavaPairRDD<String, Long> count;
        Random randomGenerator = new Random();
        count = docs
                //Reads the different occurrences for each word.
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(randomGenerator.nextInt(K), new Tuple2<>(e.getKey(), e.getValue())));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1)
                .flatMapToPair((triplet) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    for (Tuple2<String, Long> c : triplet._2()) {
                        counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        if(Pattern.matches("[a-zA-Z]+", e.getKey())){
                            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                        }else{
                            pairs.remove(new Tuple2<>(e.getKey(), e.getValue()));
                        }
                    }
                    //System.out.println(pairs);
                    return pairs.iterator();
                })
                .groupByKey()
                .sortByKey()// <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });



        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.println("Output pairs = " + count.collect());


    }


}
