package ballknn;

import java.io.*;
import java.net.URI;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

public class DistributedCache_knn_k {
	// private BufferedReader modelBR = new BufferedReader(new
	// FileReader("/home/mjiang/java/eclipse/hadoop/Target-1/data/models.txt"));
	public static class Result{
		
		private double distance=0.0;
		private String vector=null;
		Result(double distance , String vector)
		{
			this.distance=distance;
			this.vector=vector;
		}
		
		
	}
	public static class TargetMapper extends Mapper<LongWritable, Text, TextPair, Text> {

		private Path[] modelPath;
		private BufferedReader modelBR;
		private ArrayList<String> R = new ArrayList<String>();// must be initialized with memory,or nullpointer 
		private String unlabeled = new String();
		private int lengthofvector = 0;
		private int  number=0;
		private ArrayList<Result> topk=new ArrayList<Result>();
		private double radius=0.0;
		private int samplenumber=0;
		
		private final TextPair tp=new TextPair();
		
		public void setup(Context context) throws IOException,InterruptedException {
			// Configuration conf = new Configuration(); /testS 为空
			Configuration conf = context.getConfiguration(); // testS 为空
			modelPath = DistributedCache.getLocalCacheFiles(conf);
			String line;
			//String[] tokens;
			BufferedReader joinReader = new BufferedReader(new FileReader(
					modelPath[0].toString()));
			try {
				while ((line = joinReader.readLine()) != null) {
					unlabeled = line.toString();// if unlabeled is just one
				}
				String[] tt = unlabeled.split(",");
				lengthofvector = tt.length;
			} finally {
				joinReader.close();
			}
		}

		protected void  cleanup(Context context) throws IOException, InterruptedException
		{
			//context.w
			int lengthoftopk=topk.size();
			for(int i=0;i<lengthoftopk;i++)
			{
				try{
					tp.set(unlabeled,topk.get(i).distance);
					context.write(tp,new Text(topk.get(i).vector));
				}catch(Exception e)
				{
					e.getMessage();
				}
			}
			
			super.cleanup(context);
		}
		
		public void map(LongWritable key, Text value, Context context)throws IOException, InterruptedException { // here we could do
															// computation
			// 获取model的文件
			String[] line = value.toString().split(",");
			// get the value of unlabeled
			double tmp_result = 0.0;
			String[] target = unlabeled.split(",");
			try {
				for (int i = 0; i < lengthofvector; i++) {
					tmp_result += Math.pow(Double.parseDouble(line[i])- Double.parseDouble(target[i]), 2);
					
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			//tp.set(unlabeled, tmp_result);
			
			//context.write(tp, value);
			if(samplenumber<5)
			{
				topk.add(new Result(tmp_result,value.toString()));
			}else{
				double max=0.0;
				int counter=-1;
				for(int i=0;i<5;i++)
				{
					if(topk.get(i).distance>=max)
					{
						max=topk.get(i).distance;
						counter=i;
					}
					
				}
				
				if(max>tmp_result)
				{
					topk.remove(counter);//how to make sure that just remove and add one
					topk.add(new Result(tmp_result,value.toString()));
				}
			}
			samplenumber++;
			/*if(samplenumber==4000){
				for(int i=0;i<5;i++)
				{
					context.write(new DoubleWritable(topk.get(i).distance),new Text(topk.get(i).vector));
				}
			}*/
		}
	}
	
	
	public static class TextPair implements WritableComparable<TextPair> {
		  private String first = null;
		  private double second = 0.0;

		  public void set(String left, double right) {
		    first = left;
		    second = right;
		  }
		  public String getFirst() {
		    return first;
		  }
		  public double getSecond() {
		    return second;
		  }

		  @Override
		  public void readFields(DataInput in) throws IOException {
			  first=in.readUTF();
		    second = in.readDouble();
		  }
		  @Override
		  public void write(DataOutput out) throws IOException {
		    out.writeUTF(first);;
		    out.writeDouble(second);
		  }
		  @Override
		  public int hashCode() {
		    return String.valueOf(first).hashCode() + String.valueOf(second).hashCode();
		  }
		  @Override
		  public boolean equals(Object right) {
		    if (right instanceof TextPair) {
		      TextPair r = (TextPair) right;
		      return r.first == first && r.second == second;
		    } else {
		      return false;
		    }
		  }
		  //这里的代码是关键，因为对key排序时，调用的就是这个compareTo方法
		  @Override
		  public int compareTo(TextPair o) {
			  
			  
			  if(second != o.second)
			  {
				  return (int)Math.signum(second - o.second);
			  }else
			  {
				  return 0;
			  }
		    /*if (first != o.first) {
		      return Math.signum(first - o.first);
		    } else if (second != o.second) {
		      return second - o.second;
		    } else {
		      return 0;
		    }*/
		  }
		}
	
	
	public static class GroupingComparator implements RawComparator<TextPair> {
		  @Override
		  public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
		    return WritableComparator.compareBytes(b1, s1, Integer.SIZE/8, b2, s2, Integer.SIZE/8);
		  }

		  @Override
		  public int compare(TextPair o1, TextPair o2) {
		    String first1 = o1.getFirst();
		    String first2 = o2.getFirst();
		    return first1.hashCode() - first2.hashCode();
		  }
		}
	
	

	public static class combiner extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {

		}
	}

	static class TargetReducer extends Reducer<TextPair, Text, DoubleWritable, Text> {
		public void reduce(TextPair key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			//System.out.println("key.second===="+key.second);
			int k=0;
			for(Text value:values){
				context.write(new DoubleWritable(key.getSecond()), value);
				k++;
				if(k==5)
				{
					break;
				}
			}
		}
	}

	public static void main(String[] args) throws Exception {

		Job job = new Job();
		Configuration conf = job.getConfiguration();
		//conf.setBoolean("wordcount.skip.patterns", true);
		DistributedCache.addCacheFile(new Path("/home/1.txt").toUri(), conf);

		job.setJarByClass(DistributedCache_knn_k.class);

		FileInputFormat.addInputPath(job, new Path("/home/twitter.txt"));
		FileOutputFormat.setOutputPath(job, new Path("/out"));

		job.setInputFormatClass(TextInputFormat.class);
		
		
		
		/*TextInputFormat.setMinInputSplitSize(job,1024L);//设置最小分片大小
		TextInputFormat.setMaxInputSplitSize(job,2*1024*1024*10L);//设置最大分片大小 
*/
		
		
		

		job.setMapperClass(TargetMapper.class);
		job.setMapOutputKeyClass(TextPair.class);
		
		job.setGroupingComparatorClass(GroupingComparator.class);
		//job.setNumReduceTasks(0);
		job.setReducerClass(TargetReducer.class);
		
		
		job.setOutputKeyClass(DoubleWritable.class);
		job.setOutputValueClass(Text.class);

		//boolean test = conf.getBoolean("wordcount.skip.patterns", false);
		// MapContext jobConf= = new MapContext(conf, null);
		//Path[] modelPath = DistributedCache.getLocalCacheFiles(conf);

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}