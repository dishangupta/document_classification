
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM ;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author dishang
 * 
 */
public class WekaSVM {

	public class PredictionLabel {
		public double originalClass;
		public double predictedClass;
	}

	public LibSVM cSVC;
	public Instances traindata;

	/**
	 * Loads the ARFF file into memory
	 * 
	 * @param file
	 * @return
	 */
	public Instances loadARFF(String file) {
		try {
			DataSource source = new DataSource(file);
			Instances data = source.getDataSet();
			// setting class attribute if the data format does not provide this
			// information
			// For example, the XRFF format saves the class attribute
			// information as
			// well
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			System.out.println("Loading " + data.numInstances()
					+ " instances...");
			return data;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Build the SVM model and saves in serialized file
	 * 
	 * @param data
	 * @param modelFileName
	 * @param optionsString
	 */
	public void buildSVMWeka(Instances data, String modelFileName,
			String optionsString) {
		try {
			LibSVM cSVC = new LibSVM();
			cSVC.setOptions(weka.core.Utils.splitOptions(optionsString));
			cSVC.buildClassifier(data);
			// save model + header
			Vector v = new Vector();
			v.add(cSVC);
			v.add(new Instances(data, 0));
			SerializationHelper.write(modelFileName, v);
			this.cSVC = (LibSVM) v.get(0);
			this.traindata = (Instances) v.get(1);
			System.out.println("Training finished!");

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Loads in memory the saved SVM model
	 * 
	 * @param modelFileName
	 */
	public void loadSVMWekaFromModelFile(String modelFileName) {
		try {
			// read model and header
			Vector v = (Vector) SerializationHelper.read(modelFileName);
			this.cSVC = (LibSVM) v.get(0);
			this.traindata = (Instances) v.get(1);
			System.out.println("Class labels are: ");
			for (int i = 0; i < this.traindata.classAttribute().numValues(); i++) {
				System.out.println(i + " : "
						+ this.traindata.classAttribute().value(i));
			}
			System.out.println("");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Predicts the labels of each instance in test data and returns the tuple
	 * of original and predicted classes. Use this instead of testModel() when
	 * you need the predicted label for each test instance
	 * 
	 * @param testdata
	 * @return
	 */
	public List<PredictionLabel> predictLabels(Instances testdata) {
		// Taken from weka example:
		// http://weka.wikispaces.com/file/view/M5PExample.java/82917117/M5PExample.java

		List<PredictionLabel> predictions = new ArrayList<PredictionLabel>();
		double logLikelihood = 0.0;
		try {
			System.out.println("Printing probabilities and predicted class:");
			System.out.println("Prob(0)\t\t\tProb(1)\t\t\tpredicted");
			for (int i = 0; i < testdata.numInstances(); i++) {
				Instance curr = testdata.instance(i);
				// create an instance for the classifier that fits the training
				// data
				// Instances object returned here might differ slightly from the
				// one
				// used during training the classifier, e.g., different order of
				// nominal values, different number of attributes.
				Instance inst = new Instance(this.traindata.numAttributes());
				inst.setDataset(this.traindata);
				for (int n = 0; n < this.traindata.numAttributes(); n++) {
					Attribute att = testdata.attribute(this.traindata
							.attribute(n).name());
					// original attribute is also present in the current dataset
					if (att != null) {
						if (att.isNominal()) {
							// is this label also in the original data?
							// Note:
							// "numValues() > 0" is only used to avoid problems
							// with nominal
							// attributes that have 0 labels, which can easily
							// happen with
							// data loaded from a database
							if ((this.traindata.attribute(n).numValues() > 0)
									&& (att.numValues() > 0)) {
								String label = curr.stringValue(att);
								int index = this.traindata.attribute(n)
										.indexOfValue(label);
								if (index != -1)
									inst.setValue(n, index);
							}
						} else if (att.isNumeric()) {
							inst.setValue(n, curr.value(att));
						} else {
							throw new IllegalStateException(
									"Unhandled attribute type!");
						}
					}
				}

				// predict class
				double pred = this.cSVC.classifyInstance(inst);
				//System.out.println(inst.classValue() + " -> " + pred);
				double dist[] = this.cSVC.distributionForInstance(inst);
				
				if (inst.classValue() == 1.0)
					logLikelihood += Math.log(dist[1]);
				else 
					logLikelihood += Math.log(dist[0]);
				
				System.out.println(dist[0] + "\t" + dist[1] + "\t" + pred);
				PredictionLabel prediction = new PredictionLabel();
				prediction.originalClass = inst.classValue();
				prediction.predictedClass = pred;
				predictions.add(prediction);
			}

			System.out.println("\nThe average log likelihood is:" + logLikelihood/testdata.numInstances());
			System.out.println("\nPredicting finished!");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return predictions;
	}

	/**
	 * Runs the trained model on given test data and outputs a summary of
	 * results
	 * 
	 * @param testdata
	 */
	public Evaluation testModel(Instances testdata) {
		try {
			Evaluation eval = new Evaluation(this.traindata);
			eval.evaluateModel(this.cSVC, testdata);
			System.out.println(eval.toSummaryString("\nResults\n======\n",
					false)); 
			return eval;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		WekaSVM svm = new WekaSVM();
		// set the path to arff file for training
		String arffFilePath = "articles.arff";
		// set the path to saved model file (or file to be saved)
		String modelFilePath = "articles.svm.model";
		// set the weka classifier options for SVM
		String optionsString = "-S 0 -K 0 -C 2 -B";

		// load the ARFF file into memory
		Instances data = svm.loadARFF(arffFilePath);

		// build the SVM model with C-SVC, Linear Kernel and C = 1
		// This is not needed if trained model has already been saved once
		svm.buildSVMWeka(data, modelFilePath, optionsString);

		// load previously trained SVM model into memory
		svm.loadSVMWekaFromModelFile(modelFilePath);

		// Predict labels of each test instance
		svm.predictLabels(data);

		// Obtain overall evaluation summary of test data
		svm.testModel(data);
	}


}
