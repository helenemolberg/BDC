import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
//Need this import to filter out the right string variables
import java.util.regex.Pattern;

import java.util.*;
//Need this import for sorting
import static java.util.stream.Collectors.*;

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

        /*
        //Setting variable
        long numdocs;

        numdocs = docs.count();
        System.out.println("Number of documents = " + numdocs);
        */

        //Partition the the RDD in K different partitions
        docs.repartition(K);

        JavaPairRDD<String, Long> count;

        //TASK 1 - VERSION WITH DETERMINISTIC PARTITIONS
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
                        //Check if the value of Key is actually a String
                        //Remove the once that are not a String
                        if(Pattern.matches("[a-zA-Z]+", e.getKey())){
                            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                        }else{
                            pairs.remove(new Tuple2<>(e.getKey(), e.getValue()));
                        }
                    }
                    //System.out.println(pairs);
                    return pairs.iterator();
                })
                .groupByKey() // <-- REDUCE PHASE (R2)
                //Sort the keys in alphabetic order
                .sortByKey()
                .mapValues((it) -> { //need this for the groupByKey method
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });



        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.println("Output pairs = " + count.collect().toString()
                .replace("[", "") //remove the brackets from the output
                .replace("]", ""));

        //TASK 2 - VERSION WITH SPARK PARTITIONS
        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();

                })

                .mapPartitionsToPair((wc) -> {    // <-- REDUCE PHASE (R1)
                    HashMap<String, Long> counts = new HashMap<>();
                    while (wc.hasNext()){
                        Tuple2<String, Long> tuple = wc.next();
                        counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        //Check if the value of Key-place is actually a String
                        //Remove the once that are not a String
                        if(Pattern.matches("[a-zA-Z]+", e.getKey())){
                            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                        }else{
                            pairs.remove(new Tuple2<>(e.getKey(), e.getValue()));
                        }

                        //System.out.println(pairs);
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                })
                ;


        //Making a map to be able to sort the collection in count in decreasing order
        Map<String, Long> sorted = count.collectAsMap()
                .entrySet()
                .stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2,
                        LinkedHashMap::new));

        //Setting variables for the first key and value of the sorted map
        String firstKey = sorted.keySet().stream().findFirst().get();
        Long firstValue = sorted.values().stream().findFirst().get();

        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent frequent class = " + "("+ firstKey + "," + firstValue + ")");
        System.out.println("Max partition size = " + (docs.repartition(K).count() / K));
        //docs.repartition(K).count() -> give the number of lines in the document
        //Divide på K partitions -> K = 4
    }


}
