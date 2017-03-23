package jch.ml.logisticregression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LogisticRegression {

	/** the learning rate */
	private double ita;

	/** the demension of x & w */
	private int demension;
	
	/** the number of data */
	private int N;
	
	/** the weight to learn */
	private double[] weights;

	/** the number of iterations */
	private int iteration;
	
	public LogisticRegression(double ita,int iteration) 
	{
		this.ita = ita;
		this.iteration = iteration;
	}
	
	public static void main(String[] args) 
	{
		// TODO Auto-generated method stub
		double ita = 0.01;
		int iteration = 2000;
		
		LogisticRegression logistic = new LogisticRegression(ita,iteration);
		logistic.train();
		logistic.test();
	}

	public void initweights()
	{
		weights = new double[demension];
		for (int i=0; i<demension; i++) 
		{
			weights[i] = 0.0;
		}
	}
	
	public void train()
	{
		List<double[]> inputs = new ArrayList<double[]>();
		
		List<Integer> outputs = new ArrayList<Integer>();
		
		Map<List<Integer>,List<double[]>> labeldata = this.getData(new File("dataset/hw3_train"));
		for(Map.Entry<List<Integer>, List<double[]>> getentry : labeldata.entrySet())
		{
			inputs = getentry.getValue();
			outputs = getentry.getKey();
		}
		
		demension = inputs.get(0).length;
		//init weights
		initweights();

		double [] grad;
		for (int i=0; i<iteration; i++) 
		{	
//			System.out.println(iternum+","+iteration);
			grad = gradientdscent(inputs,outputs);
			updateweights(grad);
//			System.out.println("grad"+grad);
		}
		
		double ein = regclassifyError(inputs,weights,outputs);
		System.out.println("Ein: "+ein);

		System.out.println("##################################");
//		double ein = calLogiRegError(inputs,weights,outputs);
//		System.out.println("Ein: "+ein);
	}
	
	public void test()
	{
		List<double[]> inputs = new ArrayList<double[]>();
		
		List<Integer> outputs = new ArrayList<Integer>();
		
		Map<List<Integer>,List<double[]>> labeldata = this.getData(new File("dataset/hw3_test"));
		for(Map.Entry<List<Integer>, List<double[]>> getentry : labeldata.entrySet())
		{
			inputs = getentry.getValue();
			outputs = getentry.getKey();
		}

		double eout = regclassifyError(inputs,weights,outputs);
		System.out.println("Eout: "+eout);
		
//		double eout = calLogiRegError(inputs,weights,outputs);
//		System.out.println("Eout: "+eout);
	}
	
	
	public double calLogiRegError(List<double[]> inputs,double[] weights,List<Integer> outputs)
	{ 
	    double error = 0.0;  
	    for(int i = 0; i < N; i++)
	    {  
	        error += Math.log(1 + Math.exp(-outputs.get(i) * multiplicate(inputs.get(i),weights)));  
	    }  
	    return (double)error / N; 
	}  
	
	public double regclassifyError(List<double[]> inputs,double[] weights,List<Integer> outputs)
	{  
	    double error = 0.0;  
	    for(int i = 0; i < N; i++)
	    {  
	    	int predict = sign(multiplicate(inputs.get(i),weights));
	    	System.out.println(predict);
	        if( predict != outputs.get(i))  
	        {
	            error++;
	        }
	    }  
	    return (double)error / N;  
	} 
	
	public int sign(double x)
	{  //sign  
	    if(x > 0.0)
	    {
	    	return 1; 
	    }
	    else 
	    {
	    	return -1;  
	   	}
	}  
	
	public double[] gradientdscent(List<double[]> intputs,List<Integer> outputs)
	{
		N = intputs.size();
		double [] grad = new double[demension];
		for(int i=0;i<N;i++)
		{
			int y = outputs.get(i);
			double[] x = intputs.get(i);
			double wTx = multiplicate(x,weights);
			double sigmoid = sigmoid((-1.0) * (double)y * wTx);
//			System.out.println("sigmoid"+sigmoid);
			double yx = 0.0;
			for(int j=0;j<demension;j++)
			{
				grad[j] += -1.0 * sigmoid * x[j] * y;  
			} 
		}
		
		for(int j=0;j<demension;j++)
		{
			grad[j] = grad[j]/N;
		}
		
		return grad;
	}
	
	public void updateweights(double[] grad)
	{
		for (int i=0; i<demension; i++) 
		{
			weights[i] = weights[i] - (ita * grad[i]);
		}
	}
	
	public double multiplicate(double[] a,double[] b)
	{
		double multiplicated = 0.0;
		for(int i=0;i<demension;i++)  
		{
			multiplicated += a[i]*b[i];
		}
		
//		System.out.println(multiplicated);
		return multiplicated;
	}
	
	public Map<List<Integer>,List<double[]>> getData(File file)
	{
		List<Integer> labels = new ArrayList<Integer>();
		List<double[]> dataset = new ArrayList<double[]>();
		Map<List<Integer>,List<double[]>> labeldata = new HashMap<List<Integer>,List<double[]>>();
		FileReader fr;
		try 
		{
			fr = new FileReader(file);
		
			BufferedReader br = new BufferedReader(fr);
			String line = null;
//			int datanum =0;
			while ((line = br.readLine()) != null) 
			{
//				System.out.println(line);
			    
				String [] datas = line.trim().split(" ");
				
				String answer = datas[20];
//				System.out.println(answer);
//				y[datanum] = Integer.parseInt(answer);
				labels.add(Integer.parseInt(answer));
				
				double [] x = new double[datas.length-1];//4145
				
			    for(int i=0;i<datas.length-1;i++)//
			    {
			    	x[i] = Double.parseDouble(datas[i]);
			    }
//			    datanum++;
			    
			    dataset.add(x);
//			    System.out.println(labels);
			}
			
			labeldata.put(labels, dataset);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return labeldata;
	}
	
	private double sigmoid(double s) 
	{
//		System.out.println("predict"+1.0 / (1.0 + Math.exp(-s)));
		return 1.0 / (1.0 + Math.exp(-s));
	}
}
