package spark1.spark;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;








//import org.apache.commons.math3.util.MultidimensionalCounter.Iterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import java.util.List;

import scala.Tuple2;
import spark1.spark.JavaLogQuery.Stats;

public class Spark_knn {

	public static class UDF implements Serializable {

	    private final int count;
	    private final int numBytes;

	    public UDF(int count, int numBytes) {
	      this.count = count;
	      this.numBytes = numBytes;
	    }
	    public UDF merge(UDF other) {
	      return new UDF(count + other.count, numBytes + other.numBytes);
	    }

	    public String toString() {
	      return String.format("bytes=%s\tn=%s", numBytes, count);
	    }
	  }
	
	
	
	
	/*public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf=new SparkConf().setAppName("javawordcount").setMaster("spark://223.3.92.151:7077"); 
		JavaSparkContext sc=new JavaSparkContext(conf); 
		//ArrayList<Double> dest=new ArrayList<Double>(Arrays.asList(80.0,88.0,128.0,1.0));
		//dest={80,88,128,1};
		//sc.broadcast(dest);
		
		JavaRDD<String> lines = sc.textFile("hdfs://223.3.92.151:9000/user/data/test_carytesian_1");
		ArrayList<String> target=new ArrayList<String>();
		target.add("9,5,6");
		
		target.add("1,1,1");
		target.add("2,2,2");
		target.add("3,3,3");
		target.add("4,4,4");
		//target={1,2,3};
		//target.add("0,2,0,0,1,1,1,0,1,0,0,1,1,1,0.000000,0.000003,0.000000,0.000000,0.000004,0.000004,0.000004,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0,2,0,0,1,1,1,0.000000,0.000003,0.000000,0.000000,0.000002,0.000002,0.000002,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0,1,0,0,1,1,1,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0,2,0,0,1,1,1,0.0");
		target.add("1.0");
		target.add("1.0");
		target.add(1.0);
		JavaRDD<String> t=sc.parallelize(target);
		JavaPairRDD<String,String> rt=t.cartesian(lines);
		
		JavaPairRDD<String,Iterable<String>> rr=rt.mapToPair(new PairFunction<Tuple2<String,String>,String,String>(){
			@Override
			public Tuple2<String,String> call(Tuple2<String,String> t)
			{
				return new Tuple2<String,String>(t._1,t._1+"|"+t._2);
			}
			
		}).groupByKey();		
		
		//.flatMapToPair(new PairFlatMapFunction<Tuple2<Iterable<String>, Double>, String, Double>() 
		JavaPairRDD<Double,String> sim=rr.values().flatMapToPair(new PairFlatMapFunction<Iterable<String>,Double,String>(){
		//JavaPairRDD<String,Double> sim=rr.values().flatMapToPair(new PairFlatMapFunction<Iterable<String>,String,Double>(){
			@Override
			//public Iterable<Tuple2<String,Double>> call(Iterable<String> it)
			public Iterable<Tuple2<Double,String>> call(Iterable<String> it)
			{
				List<Tuple2<Double,String>> results = new ArrayList<Tuple2<Double, String>>();
				//List<Tuple2<String,Double>> results = new ArrayList<Tuple2<String, Double>>();
				Iterator<String> ir=it.iterator();//原来还有这个玩意啊  import org.apache.commons.math3.util.MultidimensionalCounter.Iterator;
				String[] tmp=null;
				String[] tmp_vector_0=null;
				String[] tmp_vector_1=null;
				double sim_vector=0.0;
				int length_of_vector=-1;
				if(ir!=null)
				{
					tmp=ir.next().toString().split("\\|");
					tmp_vector_0=tmp[0].split("\\,");
					tmp_vector_1=tmp[1].split("\\,");
					length_of_vector=tmp_vector_0.length;
					for(int i=0;i<length_of_vector;i++)
					{
						sim_vector+=Math.pow(Double.valueOf(tmp_vector_0[i])-Double.valueOf(tmp_vector_1[i]),2);
						
						System.out.println(tmp_vector_0[i]+tmp_vector_1[i]);
					}
					System.out.println(sim_vector);
					results.add(new Tuple2<Double,String>(sim_vector,tmp[0]+"|"+tmp[1]));
					//results.add(new Tuple2<String,Double>(tmp[0]+"|"+tmp[1],sim_vector));
				}
				
				while(ir.hasNext())//here we could make our improved knn come true
				{
					tmp=ir.next().toString().split("\\|");
					tmp_vector_0=tmp[0].split("\\,");
					tmp_vector_1=tmp[1].split("\\,");
					sim_vector=0.0;
					for(int i=0;i<length_of_vector;i++)
					{
						sim_vector+=Math.pow(Double.valueOf(tmp_vector_0[i])-Double.valueOf(tmp_vector_1[i]),2);
						System.out.println(tmp_vector_0[i]+tmp_vector_1[i]);
					}	
					System.out.println(sim_vector);
					results.add(new Tuple2<Double,String>(sim_vector,tmp[0]+"|"+tmp[1]));
					//results.add(new Tuple2<String,Double>(tmp[0]+"|"+tmp[1],sim_vector));
				}
				return results;
			}
		});//.sortByKey();
		
		
		
		JavaPairRDD<Double,String> res_27=sim.groupBy(new Function<Tuple2<Double,String>,Tuple2<Double,String>>(){
			@Override
			public Tuple2<Double,String> call(Tuple2<Double,String> t)
			{
				
				//<Tuple2<Double,String>,Iterable<Tuple2<Double,String>>>
				
				
				List<Tuple2<Double,String>> res= new ArrayList<Tuple2<Double, String>>();
				return new Tuple2<Double,String>(t._1,t._2);
			}
			});
		List<Tuple2<Double,String>> re=sim.collect();
		for(Tuple2<Double,String> t_t:re)
		{
			System.out.println(t_t._1() + "\t" + t_t._2());
		}
		
		
	}*/
	
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		//JavaRDD<String> lines = sc.textFile("hdfs://master:8020/home/3.txt");
		//SparkConf conf=new SparkConf().setAppName("spark knn").setMaster("spark://223.3.92.151:7077");
		
		SparkConf conf=new SparkConf().setAppName("spark knn").setMaster("yarn-cluster");
		
		JavaSparkContext sc=new JavaSparkContext(conf); 
		//ArrayList<Double> dest=new ArrayList<Double>(Arrays.asList(80.0,88.0,128.0,1.0));
		//dest={80,88,128,1};
		//sc.broadcast(dest);
		//System
		
		if(args.length<1)
		{
			System.out.println("please determined the target");
			return;
		}
		JavaRDD<String> lines = sc.textFile("hdfs://master:8020"+args[0]);
		//JavaRDD<String> lines = sc.textFile("hdfs://master:8020/home/3.txt");
		JavaRDD<String> t = sc.textFile("hdfs://master:8020/user/hdfs/lpp/data/knn_join/sift_40w_.txt");
		//JavaRDD<String> t = sc.textFile("hdfs://master:8020/home/twitter20.txt");
		//JavaRDD<String> lines = sc.textFile("hdfs://223.3.92.151:9000/user/data/test_carytesian_1");
		/*ArrayList<String> target=new ArrayList<String>();
		target.add("9,5,6");
		
		target.add("1,1,1");
		target.add("2,2,2");
		target.add("3,3,3");
		target.add("4,4,4");
		//target={1,2,3};
		//target.add("0,2,0,0,1,1,1,0,1,0,0,1,1,1,0.000000,0.000003,0.000000,0.000000,0.000004,0.000004,0.000004,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0,2,0,0,1,1,1,0.000000,0.000003,0.000000,0.000000,0.000002,0.000002,0.000002,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0,1,0,0,1,1,1,0.000000,1.000000,0.000000,0.000000,1.000000,1.000000,1.000000,0,2,0,0,1,1,1,0.0");
		//target.add("1.0");
		//target.add("1.0");
		//target.add(1.0);
		JavaRDD<String> t=sc.parallelize(target);*/
		
		
		
		//JavaPairRDD<String,String> rt=t.cartesian(lines);
		JavaPairRDD<String,String> rt=lines.cartesian(t);
		JavaPairRDD<String,Iterable<String>> rr=rt.mapToPair(new PairFunction<Tuple2<String,String>,String,String>(){
			@Override
			public Tuple2<String,String> call(Tuple2<String,String> t)
			{
				return new Tuple2<String,String>(t._1,t._1+"|"+t._2);
			}
			
		}).groupByKey();		
		
		//.flatMapToPair(new PairFlatMapFunction<Tuple2<Iterable<String>, Double>, String, Double>() 
		JavaPairRDD<Double,String> sim=rr.values().flatMapToPair(new PairFlatMapFunction<Iterable<String>,Double,String>(){
		//JavaPairRDD<String,Double> sim=rr.values().flatMapToPair(new PairFlatMapFunction<Iterable<String>,String,Double>(){
			@Override
			//public Iterable<Tuple2<String,Double>> call(Iterable<String> it)
			public Iterable<Tuple2<Double,String>> call(Iterable<String> it)
			{
				List<Tuple2<Double,String>> results = new ArrayList<Tuple2<Double, String>>();
				//List<Double> distance=new ArrayList<Double>();
				int k=5;
				int counter=-1;
				//List<Tuple2<String,Double>> results = new ArrayList<Tuple2<String, Double>>();
				Iterator<String> ir=it.iterator();//原来还有这个玩意啊  import org.apache.commons.math3.util.MultidimensionalCounter.Iterator;
				String[] tmp=null;
				String[] tmp_vector_0=null;
				String[] tmp_vector_1=null;
				double sim_vector=0.0;
				int length_of_vector=0;
				double max=0.0;
				int j=0;
				//if(ir!=null)
				//{
				while(ir.hasNext()&&j<k)
				{
					tmp=ir.next().toString().split("\\|");
					tmp_vector_0=tmp[0].split("\\,");
					tmp_vector_1=tmp[1].split("\\,");
					length_of_vector=tmp_vector_0.length;
					sim_vector=0.0;
					for(int i=0;i<length_of_vector;i++)
					{
						sim_vector+=Math.pow(Double.valueOf(tmp_vector_0[i])-Double.valueOf(tmp_vector_1[i]),2);
						
						//System.out.println(tmp_vector_0[i]+tmp_vector_1[i]);
					}
					//System.out.println(sim_vector);
					
					
					if(max<sim_vector)
					{
						max=sim_vector;
						
						counter=j;
					}
					results.add(new Tuple2<Double,String>(sim_vector,tmp[0]+"|"+tmp[1]));
					j++;
					
					//counter++;
					//results.add(new Tuple2<String,Double>(tmp[0]+"|"+tmp[1],sim_vector));
				}
				//}
				while(ir.hasNext())//here we could make our improved knn come true
				{
					tmp=ir.next().toString().split("\\|");
					tmp_vector_0=tmp[0].split("\\,");
					tmp_vector_1=tmp[1].split("\\,");
					sim_vector=0.0;
					for(int i=0;i<length_of_vector;i++)
					{
						sim_vector+=Math.pow(Double.valueOf(tmp_vector_0[i])-Double.valueOf(tmp_vector_1[i]),2);
						//System.out.println(tmp_vector_0[i]+tmp_vector_1[i]);
					}	
					//System.out.println(sim_vector);
					
					if(sim_vector<max)
					{
						results.remove(counter);
						results.add(new Tuple2<Double,String>(sim_vector,tmp[0]+"|"+tmp[1]));
						max=0.0;
						for(int p=0;p<k;p++)
						{
							if(results.get(p)._1>max)
							{
								max=results.get(p)._1;
								counter=p;
							}
						}
					}
					//results.add(new Tuple2<String,Double>(tmp[0]+"|"+tmp[1],sim_vector));
				}
				return results;
			}
		});//.sortByKey();
		
		
		
		/*JavaPairRDD<Double,String> res_27=sim.groupBy(new Function<Tuple2<Double,String>,Tuple2<Double,String>>(){
			@Override
			public Tuple2<Double,String> call(Tuple2<Double,String> t)
			{
				
				//<Tuple2<Double,String>,Iterable<Tuple2<Double,String>>>
				
				
				List<Tuple2<Double,String>> res= new ArrayList<Tuple2<Double, String>>();
				return new Tuple2<Double,String>(t._1,t._2);
			}
			});*/
		//List<Tuple2<Double,String>> re=sim.collect();
		sim.saveAsTextFile("hdfs://master:8020/user/hdfs/lpp/result");
		
		/*for(Tuple2<Double,String> t_t:re)
		{
			System.out.println(t_t._1() + "\t" + t_t._2());
		}*/
		
		
	}


}
