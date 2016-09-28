import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
		 
public class TagText {
	
	public static void main(String[] args) throws IOException,
			ClassNotFoundException {
		
		// Initialize the tagger
		MaxentTagger tagger = new MaxentTagger("models\\left3words-wsj-0-18.tagger");
		
		// Input and Output Directory
		String inputDirectory = "..\\Documents";
		String outputDirectory = "..\\Documents_POS";
		
		int numFiles = 1000;
		
		for (int i = 0; i < numFiles; i++) {
			// Input and Output File
			System.out.println("File: " + (i+1));
			
			String inputFile = inputDirectory + "\\article" + (i+1);
			String outputFile = outputDirectory + "\\article" + (i+1) + "_pos";
						
		
			try {
				File inFile = new File(inputFile);
				File outFile = new File(outputFile);
				
				BufferedReader reader = new BufferedReader(new FileReader(inFile));
				BufferedWriter writer = new BufferedWriter(new FileWriter(outFile, true));
								
				String line = "";
				
				List<String> words = new ArrayList<String>();
				
				while((line = reader.readLine()) != null) {
					
					// First <S> marker
					if (words.isEmpty() && line.matches("<S>"))
						continue;
					
					// Extract sentence
					if (line.matches("<S>")) {
						
						List<Word> sentence = new ArrayList<Word>();
						
						for (String s: words) {
							Word word = new Word();
							word.setWord(s);
							sentence.add(word);
						}
						
						// Tag sentence
						List<TaggedWord> taggedList = tagger.tagSentence(sentence);
						
						writer.write("<S>" + "\n");
						for (TaggedWord tagWord: taggedList)
							writer.write(tagWord.tag() + "\n");
							
						words.clear();
						continue;
					}
					words.add(line);
				}
				
				List<Word> sentence = new ArrayList<Word>();
				
				for (String s: words) {
					Word word = new Word();
					word.setWord(s);
					sentence.add(word);
				}
				
				// Tag sentence
				List<TaggedWord> taggedList = tagger.tagSentence(sentence);
				
				writer.write("<S>" + "\n");
				for (TaggedWord tagWord: taggedList)
					writer.write(tagWord.tag() + "\n");
					
				words.clear();
				
				reader.close();
				writer.close();
			
			} catch (IOException e) {
				System.out.println("File Not Found!");
			} 
		} 
	}
}
